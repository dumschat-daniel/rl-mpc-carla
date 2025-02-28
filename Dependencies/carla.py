import settings
from Dependencies import STOP, RGB_camera, Preview_camera, Collision_detector, Depth_camera, IMU_Sensor, Lidar, Radar, Obstacle_detector, Lane_invasion_detector, Semantic_segmentation_camera, Instance_segmentation_camera, GNSS_sensor
import math
import random
import sys
import glob
import os
try:
    sys.path.append(glob.glob(settings.CARLA_PATH + f'/PythonAPI/carla/dist/carla-*{sys.version_info.major}.{sys.version_info.minor}-{"win-amd64" if os.name == "nt" else "linux-x86_64"}.egg')[0])
except IndexError:
    pass
import carla
import time
import psutil
import subprocess
from queue import Queue
from dataclasses import dataclass
import re
from multiprocessing import Value, Event
import numpy as np


# ACTIONS are discrete and only relevant for dqn.
@dataclass(frozen=True)
class ACTIONS:
    forward_slow: int = 0
    forward_medium: int = 1
    left_slow: int = 2
    left_medium: int = 3
    right_slow: int = 4
    right_medium: int = 5
    brake: int = 6
    no_action: int = 7

ACTION_CONTROL = {
    0: [0.3, 0, 0],
    1: [0.6, 0, 0],
    2: [0.3, 0, -0.3],
    3: [0.3, 0, -0.6],
    4: [0.3, 0, 0.3],
    5: [0.3, 0, 0.6],
    6: [0, 0.5, 0],
    7: None,
}

ACTIONS_NAMES = {
    ACTIONS.forward_slow: 'forward_slow',
    ACTIONS.forward_medium: 'forward_medium',
    ACTIONS.left_slow: 'left_slow',
    ACTIONS.left_medium: 'left_medium',
    ACTIONS.right_slow: 'right_slow',
    ACTIONS.right_medium: 'right_medium',
    ACTIONS.no_action: 'no_action',
}

CAM_DICT = {
    'rgb': RGB_camera,
    'depth' : Depth_camera,
    'semseg' : Semantic_segmentation_camera,
    'inseg' : Instance_segmentation_camera
}


class CarlaEnv:
    
    action_space_size = len(settings.ACTIONS)

    def __init__(self, carla_instance, sync_mode, env_settings_cond, step_cond, rl_algorithm='dqn', seconds_per_episode=None, steps_per_episode=None, synchronizer=None, testing=False):
        """Communication with the Carla Server and implements the reset and step methods from RL."""

        # synchronization between env Settings and Agents
        self.sync_mode = sync_mode 
        self.synchronizer = synchronizer 
        self.env_settings_cond = env_settings_cond 
        self.step_cond = step_cond

        self.client = carla.Client(*settings.CARLA_HOSTS[carla_instance][:2])
        self.client.set_timeout(5.0)
        
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.find(settings.VEHICLE)
        
        
        self.rl_algorithm = rl_algorithm
        self.actor_list = []
        self.front_camera = None
        self.preview_camera = None
        
        self.seconds_per_episode = seconds_per_episode
        self.steps_per_episode = steps_per_episode
        
        self.testing = testing
        self.preview_camera_enabled = False

        self.actions = [getattr(ACTIONS, action) for action in settings.ACTIONS]

        
    
    def reset(self, sp=None):
        """setup for episode. Spawns Vehicle, sets up Sensors and returns first Episode State"""
        # synchronization
        if not self.testing and self.sync_mode and settings.AGENTS > 1:
            self.synchronizer.reset()


        self.actor_list = []
        spawn_start = time.perf_counter()
        sensor_data = {}
        self.additional_data = {} # dict containing additional data for logging
        self.step_data = {}

        # spawning vehicle
        spawn_points = self.world.get_map().get_spawn_points()
        while True:
            try:
                if sp is None:
                    transform = random.choice(self.world.get_map().get_spawn_points())
                else:
                    transform = self.world.get_map().get_spawn_points()[sp]

                self.cur_sp = spawn_points.index(transform)
                self.vehicle = self.world.spawn_actor(self.vehicle_bp, transform)
                break
            except:
                time.sleep(0.001)

            if time.perf_counter() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')

        self.actor_list.append(self.vehicle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        

        # synchronization, it takes some time for our vehicle to accept input
        if not self.sync_mode:
            time.sleep(settings.SETUP_TIME)
        elif self.testing or settings.AGENTS == 1:
            ticks = 0
            fixed_delta_seconds = self.world.get_settings().fixed_delta_seconds
            while ticks < (settings.SETUP_TIME / fixed_delta_seconds):
                self.world.tick()
                ticks += 1
        else: 
            fixed_delta_seconds = self.world.get_settings().fixed_delta_seconds
            current_frame = self.world.get_snapshot().frame
            target_frame = current_frame + math.ceil(settings.SETUP_TIME / fixed_delta_seconds)
            while self.world.get_snapshot().frame < target_frame:
                if self.synchronizer.resetting_agents.value == self.synchronizer.total_agents and self.world.get_snapshot().frame == current_frame:
                    self.world.tick()
                current_frame = self.world.get_snapshot().frame

        # create and attach Sensors
        self.create_sensors()

        # synchronization
        if self.sync_mode:
            self.world.tick()

        # extra setup time for testing to make sure vehicle is stable and testing is as accurate as possible. Normal Setup time is enough for training
        if self.testing:
            for _ in range(50):
                self.world.tick()
        # wait for image data which means everything setup correctly
        while self.front_camera.image is None or (self.preview_camera_enabled and self.preview_camera.image is None):
            time.sleep(0.01)

        
        sensor_data, agent_view = self.get_sensor_data() 

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))   #disengage brakes
  
        self.episode_start = time.perf_counter()
        self.cur_step = 0
        
        # synchronization
        if not self.testing and self.sync_mode and settings.AGENTS > 1:
            self.synchronizer.reset_finished()

        return sensor_data, 0, agent_view


    def step(self, action):
        """Executes Actions, gives Rewards and returns the new State"""
        # Monitor if carla stopped sending images for longer than a second. If yes - it broke
        if time.perf_counter() > self.front_camera.last_cam_update + 1:
            raise Exception('Missing updates from Carla')
        self.cur_step += 1
        last_vehicle = False
        sensor_data = {}
        smoothened_action = None

        throttle, steer, brake = 0, 0, 0
        # Extract Action based on Algorithm. Dqn provides an int that corresponds to an Action and ddpg/td3 give an acceleration and a steering value (-1 to 1)
        if self.rl_algorithm == 'dqn':
            if settings.SMOOTH_ACTIONS == True:
                throttle, brake, steer = self.smooth_action(action)
            else:
                throttle = ACTION_CONTROL[self.actions[action]][0]
                brake = ACTION_CONTROL[self.actions[action]][1]
                steer = ACTION_CONTROL[self.actions[action]][2]
        else:
            throttle = float(max(action[0], 0))
            brake = float(abs(min(action[0], 0)))
            steer = float(action[1])


        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

        # synchronization
        if self.sync_mode:
            if self.testing or settings.AGENTS == 1:
                with self.env_settings_cond:
                    self.env_settings_cond.notify()
                with self.step_cond:
                    self.step_cond.wait()
                last_vehicle = True

            else:
                self.synchronizer.step()
                # Wait for all agents to reach this point
                self.synchronizer.event.wait()
                with self.synchronizer.stepping_agents.get_lock():
                    self.synchronizer.stepping_agents.value -= 1
                    if self.synchronizer.stepping_agents.value == 0:
                        self.synchronizer.event.clear() 
                        last_vehicle = True
                        with self.env_settings_cond:
                            self.env_settings_cond.notify()
                        with self.step_cond:
                            self.step_cond.wait()
            
            if not last_vehicle:        
                with self.step_cond:
                    self.step_cond.wait() 
        
        
        # get rewards and sensor data
        reward, done = self.get_rewards(action)

        sensor_data, agent_view = self.get_sensor_data(action, smoothened_action)
        
        return sensor_data, reward, done, agent_view, self.additional_data

    

    def get_sensor_data(self, action=None, smoothened_action=None):
        """return data from all attached sensors"""
        model_inputs = settings.MODEL_INPUTS
        sensor_data = {}

        #if 'front_camera' in model_settings:
        agent_view = self.front_camera.image
        
        if model_inputs['front_camera']:
            sensor_data['front_camera'] = self.front_camera.image


        if model_inputs['lidar']:
            sensor_data['lidar'] = self.lidar.data

        if model_inputs['collision']:
            sensor_data['collision'] = len(self.collision_detector.collision_hist) != 0
            self.collision_detector.collision_hist = []

        if model_inputs['lane_invasion']:
            sensor_data['lane_invasion'] = len(self.lane_invasion_detector.data) != 0
            self.lane_invasion_detector.data = []

        if model_inputs['speed']:
            v = self.vehicle.get_velocity()
            sensor_data['speed'] = round(math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),2)

        if model_inputs['relative_pos']:
            navigation_data = self.gnss_sensor.calculate_waypoint_alignment_errors(self.vehicle)
            sensor_data['navigation'] = navigation_data if model_inputs['relative_orientation'] else navigation_data[:-1]

        if model_inputs['last_action']:
            sensor_data['last_action'] = action
        
        if model_inputs['last_agent_input']:
            sensor_data['last_agent_input'] = smoothened_action
        
        if model_inputs['distance_to_lane_center']:
            lane_center_data = self.gnss_sensor.calculate_lane_alignment_errors(self.vehicle)
            sensor_data['lane_center'] = lane_center_data if model_inputs['orientation_difference_to_lane_center'] else lane_center_data[:-1]

        if model_inputs['close_vehicles']:
            sensor_data['close_vehicles'] = self.gnss_sensor.get_close_vehicles()

        if model_inputs['acceleration']:
            sensor_data['acceleration'] = self.vehicle.get_acceleration()

        if model_inputs['yaw_angle']:
            sensor_data['yaw_angle'] = math.radians(self.vehicle.transform.rotation.yaw)

        if model_inputs['jerk_rate']:
            ...
        if model_inputs['traffic_light_state']:
            sensor_data['traffic_light_state'] = self.gnss_sensor.get_traffic_light_state()

        return sensor_data, agent_view
    

    
    def get_rewards(self, action=None):
        """calculate the Rewards for the Step and checks for Terminal Conditions"""
        done = False
        reward = -settings.MOVE_PENALTY if settings.REWARD_FUNCTION_METRICS['move'] else 0
        additional_rewards = 0


        if settings.REWARD_FUNCTION_METRICS['collision'] and len(self.collision_detector.collision_hist) != 0:
            done = True
            reward = -settings.COLLISION_PENALTY
            self.additional_data['episode_end_reason'] = 0
            self.additional_data.setdefault('collision_rewards', 0)
            self.additional_data['collision_rewards'] += reward
            return reward, done


        if settings.REWARD_FUNCTION_METRICS['waypoint_reached']:
            navigation_reward, is_done = self.gnss_sensor.is_vehicle_in_waypoint_radius(self.vehicle)
            additional_rewards += navigation_reward
            self.additional_data.setdefault('navigation_rewards', 0)
            self.additional_data['navigation_rewards'] += navigation_reward
            
            if is_done:
                done = True
                self.additional_data['episode_end_reason'] = 1
                return navigation_reward, done
            
            

        if settings.REWARD_FUNCTION_METRICS['lane_invasion']:
            lane_invasion_data = self.lane_invasion_detector.data
            for filter_marking, lane_invasion_penalty in settings.LANE_INVASION_FILTER:
                if lane_invasion_data == filter_marking:
                    reward += lane_invasion_penalty
                    

        if settings.REWARD_FUNCTION_METRICS['lane_center']:
            lane_center_reward, is_done = self.gnss_sensor.get_lane_center_rewards(self.vehicle)
            reward += lane_center_reward
            self.additional_data.setdefault('lane_center_rewards', 0)
            self.additional_data['lane_center_rewards'] += lane_center_reward
            
            if is_done and not self.testing:
                done = True
                self.additional_data['episode_end_reason'] = 2
                return lane_center_reward, done
            
            
        if settings.REWARD_FUNCTION_METRICS['speed']:    
            v = self.vehicle.get_velocity()
            ms = np.linalg.norm([v.x, v.y, v.z]) 
            kmh = 3.6 * ms
            if settings.WEIGHT_REWARDS_WITH_SPEED == 'discrete':
                speed_reward = -settings.SPEED_MAX_PENALTY if kmh < 50 else settings.SPEED_MAX_REWARD

            elif settings.WEIGHT_REWARDS_WITH_SPEED == 'linear':
                speed_reward = kmh * (settings.SPEED_MAX_REWARD + settings.SPEED_MAX_PENALTY) / settings.MAX_SPEED -settings.SPEED_MAX_PENALTY

            elif settings.WEIGHT_REWARDS_WITH_SPEED == 'quadratic':
                speed_reward = (kmh / settings.MAX_SPEED) ** 1.3 * (settings.SPEED_MAX_REWARD + settings.SPEED_MAX_PENALTY) - settings.SPEED_MAX_PENALTY

            elif settings.WEIGHT_REWARDS_WITH_SPEED == 'area':
                if settings.MIN_SPEED and settings.MIN_SPEED <= kmh <= settings.MAX_SPEED:
                    speed_reward = settings.SPEED_MAX_REWARD
                elif settings.MAX_SPEED * 0.9 <= kmh <= settings.MAX_SPEED * 1.1:
                    speed_reward = settings.SPEED_MAX_REWARD
                else:
                    target_speed = (settings.MIN_SPEED + settings.MAX_SPEED) /2
                    deviation = abs((target_speed - kmh) / target_speed * settings.SPEED_MAX_PENALTY)
                    speed_reward = -max(deviation, settings.SPEED_MAX_PENALTY) 
            
            
            reward += speed_reward


            self.additional_data.setdefault('speed_rewards', 0)
            self.additional_data['speed_rewards'] += speed_reward
            
            

            self.gnss_sensor.speeds.append(ms)
            

        if not self.testing and not self.sync_mode:
            if self.episode_start + self.seconds_per_episode.value > time.perf_counter():
                done = True
                self.additional_data['episode_end_reason'] = 3

            if settings.REWARD_FUNCTION_METRICS['progress'] and not done:
                reward *= (time.perf_counter() - self.episode_start) / self.seconds_per_episode.value

        if not self.testing and self.sync_mode:
            
            if self.cur_step >= self.steps_per_episode.value:
                done = True
                self.additional_data['episode_end_reason'] = 3
                
            if settings.REWARD_FUNCTION_METRICS['progress'] and not done:
                reward *= self.cur_step / self.steps_per_episode.value

        if settings.REWARD_FUNCTION_METRICS['jerk']:
            ...
                
        reward += additional_rewards

        # Additional Penaltys to overcome local minima
        if 'speed_slow' in settings.ADDITIONAL_PENALTYS and kmh < 20 and action[0] < 0.5:
            reward -= 2
        if 'speed_fast' in settings.ADDITIONAL_PENALTYS and kmh > 30 and action[0] > 0.3:
            reward -= 2
        if 'steering' in settings.ADDITIONAL_PENALTYS and abs(action[1]) > 0.5:
            reward -= 2

        if settings.REWARD_RANGE:
            reward = min(max(reward, settings.REWARD_RANGE[0]), settings.REWARD_RANGE[1])
  
        return reward, done


    def get_vehicle_state(self):
        """get Orientation and Velocity of the Vehicle"""
        transform = self.vehicle.get_transform()
        rotation = transform.rotation
        orientation = {
            "roll": math.radians(rotation.roll),
            "pitch": math.radians(rotation.pitch),
            "yaw": math.radians(rotation.yaw),

        }
        velocity = self.vehicle.get_velocity()

        velocity = math.sqrt(velocity.x**2 + velocity.y ** 2 + velocity.z ** 2)
        
        vehicle_status = {
            "orientation": orientation,
            "velocity": velocity
        }

        return vehicle_status
    

    def calculate_mpc_state(self, s_current):
        """calculate the current Vehicle State for MPC"""
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        x_vehicle = location.x
        y_vehicle = location.y
        yaw_vehicle = np.radians(rotation.yaw)  

        # reference trajectory
        x_ref = self.gnss_sensor.spline_x(s_current)
        y_ref = self.gnss_sensor.spline_y(s_current)

        
        dx_ref = self.gnss_sensor.spline_x(s_current, 1)
        dy_ref = self.gnss_sensor.spline_y(s_current, 1)


        # Calculate next state
        ref_heading = np.arctan2(dy_ref, dx_ref)


        e_y = ((x_vehicle - x_ref) * np.sin(ref_heading) -
                        (y_vehicle - y_ref) * np.cos(ref_heading))

        e_psi = yaw_vehicle - ref_heading

        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))

        velocity = self.vehicle.get_velocity()
        v = np.sqrt(velocity.x**2 + velocity.y**2)

        curvature = self.gnss_sensor.compute_curvature(s_current)

        s_dot = v * np.cos(e_psi) / (1 - curvature * e_y)  # Calculate s_dot
        s_next = s_current + s_dot * settings.FIXED_DELTA_SEC  # Update s

        state = [s_next, e_y, e_psi, v]
        return state



    def create_sensors(self):
        """creates Sensors at the start of Episode"""
        # front camera is also used for agent view
        self.front_camera = CAM_DICT.get(settings.FRONT_CAM_TYPE, None)(self.world, self.blueprint_library, self.vehicle, settings.FRONT_CAM_SETTINGS, self.frametimes, self.testing)
        self.actor_list.append(self.front_camera)

        self.collision_detector = Collision_detector(self.world, self.blueprint_library, self.vehicle)
        self.actor_list.append(self.collision_detector)
        
        self.rgb_cam = None
        self.depth_cam = None
        self.imu_sensor = None
        self.lane_invasion_detector = None
        self.lidar = None
        self.radar = None
        self.obstacle_detector = None
        self.semantic_segmentation = None
        self.instance_segmentation = None

        if self.preview_camera_enabled is not False:
            self.preview_camera = Preview_camera(self.world, self.blueprint_library, self.vehicle, self.preview_camera_enabled)
            self.actor_list.append(self.preview_camera)

        if settings.REWARD_FUNCTION_METRICS['waypoint_reached'] or settings.MODEL_INPUTS['relative_pos']:

            self.gnss_sensor = GNSS_sensor(self.world, self.blueprint_library, self.vehicle, settings.GNSS_SENSOR_SETTINGS)
            self.gnss_sensor.generate_route(self.world, self.vehicle)
            self.actor_list.append(self.gnss_sensor)

            
        if settings.REWARD_FUNCTION_METRICS['imu']:
            self.imu_sensor = IMU_Sensor(self.world, self.blueprint_library, self.vehicle, settings.IMU_SETTINGS)
            self.actor_list.append(self.imu_sensor)
        if settings.REWARD_FUNCTION_METRICS['lane_invasion'] or settings.MODEL_INPUTS['lane_invasion']:
            self.lane_invasion_detector = Lane_invasion_detector(self.world, self.blueprint_library, self.vehicle, settings.LANE_INVASION_SETTINGS)
            self.actor_list.append(self.lane_invasion_detector)
        if settings.MODEL_INPUTS['lidar']:
            self.lidar = Lidar(self.world, self.blueprint_library, self.vehicle, settings.LIDAR_SETTINGS)
            self.actor_list.append(self.lidar)
        if settings.MODEL_INPUTS['radar']:
            self.radar = Radar(self.world, self.blueprint_library, self.vehicle, settings.RADAR_SETTINGS)
            self.actor_list.append(self.radar)
        if settings.MODEL_INPUTS['obstacle']:
            self.obstacle_detector = Obstacle_detector(self.world, self.blueprint_library, self.vehicle, settings.OBSTACLE_DETECTOR_SETTINGS)
            self.actor_list.append(self.obstacle_detector)
          


    # not implemented yet, only for dqn
    def smooth_action(self, action):
        return action


    
    def destroy_actors(self):
        """cleans up Environment at the end of the Episode"""
        for actor in self.actor_list:
            if hasattr(actor, 'is_alive') and actor.is_alive:
                actor.destroy()
            else:
                actor.destroy()
                
        self.actor_list = []



def set_sync_mode(carla_host, fixed_delta_sec):
    """Set Sync Mode and dt"""
    client = carla.Client(carla_host[0], carla_host[1])
    client.set_timeout(5.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_sec
    world.apply_settings(settings)


def get_fixed_delta_seconds():
    """return current dt"""
    client = carla.Client(settings.CARLA_HOSTS[0][:2])
    client.set_timeout(5.0)
    world = client.get_world()
    settings = world.get_settings()
    return settings.fixed_delta_seconds


class Synchronizer():
    """Synchronizes multi Agents and the Environment in Sync mode"""
    def __init__(self):
        self.event = Event()
        # keeps track of what agents are doing, resetting agents shouldn't block the Steps
        self.resetting_agents = Value('i', 0)
        self.stepping_agents = Value('i', 0)
        self.total_agents = settings.AGENTS
        
    def reset(self):
        with self.resetting_agents.get_lock():
            self.resetting_agents.value += 1
            self.check_event()
    
    def reset_finished(self):
        with self.resetting_agents.get_lock():
            self.resetting_agents.value -= 1
            
    def step(self):
        with self.stepping_agents.get_lock():
            self.stepping_agents.value += 1
            self.check_event()
            
    def check_event(self):
        if self.resetting_agents.value + self.stepping_agents.value >= self.total_agents and self.stepping_agents.value > 0:
            self.event.set()
        else:
            return  



'''
CARLA Setup Methods
'''
operating_system = 'windows' if os.name == 'nt' else 'linux'

def get_binary():
    return 'CarlaUE4.exe' if operating_system == 'windows' else 'CarlaUE4.sh'


def get_exec_command():
    binary = get_binary()
    exec_command = binary if operating_system == 'windows' else ('./' + binary)

    return exec_command


def check_carla_port(port):
    """check if chosen CARLA port is occupied"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            connections = proc.connections(kind='inet') 

            for conn in connections:
                if conn.laddr.port == port:  
                    if "CarlaUE4" in proc.name():
                        return 0  

                    if settings.FREE_OCCUPIED_PORT:
                        proc.terminate() 
                        proc.wait()  
                        print(f"Process {proc.name()} terminated successfully.")
                        return 0 
                    else:
                        return -1  
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
           
            continue

def kill_carla_processes():
    """Kills CARLA process if still running on Start"""
    binary = get_binary()

    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            try:
                process.terminate()
            except:
                pass

    still_alive = []
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            still_alive.append(process)

    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)

def setup_carla_hosts(hosts, testing=False, max_retries=float("inf")):
    """Starts CARLA Hosts, set's up Map and Layers"""
    for process_no in range(1 if testing else settings.CARLA_HOSTS_NO):
        retries = 0
        while True:
            retries += 1
            try:
                client = carla.Client(*hosts[process_no][:2])
                client.set_timeout(5.0)
                
                if len(hosts[process_no]) < 3 or not hosts[process_no][2]:
                    break
                map_choice = hosts[process_no][2]
                available_maps = [x.split('/')[-1] for x in client.get_available_maps()]
                if map_choice not in available_maps:
                    map_choice = random.choice([map.split('/')[-1] for map in available_maps])
                
                if settings.MAP_LAYER_GROUND_ONLY and map_choice == hosts[process_no][2]:
                    world = client.load_world(map_choice, map_layers=carla.MapLayer.Ground)
                else:
                    world = client.get_world()
                cur_map = world.get_map().name.split('/')[-1]
                            
                if cur_map != map_choice:
                    if settings.MAP_LAYER_GROUND_ONLY:
                        client.load_world(map_choice, map_layers=carla.MapLayer.Ground)
                    else:
                        client.load_world(map_choice)
                    while True:
                        try:
                            while world.get_map().name.split('/')[-1] != map_choice:
                                time.sleep(1.0)
                                retries += 1
                                if retries >= 5:
                                    raise Exception('Couldn\'t change map [1]')
                            break           
                        except Exception as e:
                            print(e)
                            time.sleep(1.0)
                        retries += 1
                        if retries >= max_retries:
                            raise Exception('Couldn\'t change map [2]')
                break
            except Exception as e:
                time.sleep(0.1)

            retries += 1
            if retries >= max_retries:
                break


def start(testing=False):
    """Entry Point for CARLA Simulation. Starts the Server"""

    if settings.CARLA_HOSTS_TYPE == "local":
        kill_carla_processes()

        amount_of_hosts = 1 if testing else settings.CARLA_HOSTS_NO 
        hosts = settings.TESTING_CARLA_HOST if testing else settings.CARLA_HOSTS 

        for process_no in range(amount_of_hosts):
            if check_carla_port(hosts[process_no][1]) == -1:
                print(f"Port {hosts[process_no][1]} is occupied. If you want to forcfully kill the process using that port set FORCE_PORT_IF_TAKEN=True")
                return -1
        
        for process_no in range(amount_of_hosts):
            command = [get_exec_command(), f' -carla-rpc-port={hosts[process_no][1]}']
            if hosts[process_no][3]:
                command.extend(['-benchmark', '-fps=' + str(hosts[process_no][3])])
            if hosts[process_no][4]:
                command.extend([f'-quality-level={hosts[process_no][4]}'])
            if hosts[process_no][5]:
                command.extend([f'-RenderOffScreen'])
            subprocess.Popen(command, cwd=settings.CARLA_PATH, shell=True)
        
        time.sleep(2) 
    setup_carla_hosts(hosts, testing) 
 

def restart(testing=False):
    setup_carla_hosts(testing, max_retries=60)


'''
CARLA Weather Methods from the CARLA files
'''

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)


    def set_new_weather(self, weather):
        self.weather = weather
        
    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)
    
@dataclass(frozen=True)
class CARLA_SETTINGS_STATE:
    starting = 0
    working = 1
    restarting = 2
    finished = 3
    error = 4


# Carla settings state messages
CARLA_SETTINGS_STATE_MESSAGE = {
    0: 'STARTING',
    1: 'WORKING',
    2: 'RESTARING',
    3: 'FINISHED',
    4: 'ERROR',
}


def find_weather_preset(weather_choice):
    """Set a Weather Preset"""
    weather_presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    for preset in weather_presets:
        if preset == weather_choice:
            return getattr(carla.WeatherParameters, preset)
    return None
    


class CarlaEnvSettings:
    """Responsible for Weather and NPCS"""
    def __init__(self, process_no, agent_pauses, sync_mode, env_settings_cond, step_cond, stop=None, car_npcs=[0, 0], stats=[0., 0., 0., 0., 0., 0.]):

        self.speed_factor = settings.SPEED_FACTOR # relevant for async mode to keep it at a certain tick rate
                
        self.process_no = process_no

        self.weather = None
        self.spawned_car_npcs = {}

        self.stats = stats

        self.restart = False

        self.car_npcs = car_npcs

        self.state = CARLA_SETTINGS_STATE.starting

        self.stop = stop

        self.collisions = Queue()

        self.world_name = None

        # Map Rotation
        self.rotate_map_every = None if not settings.ROTATE_MAP_EVERY else settings.ROTATE_MAP_EVERY * 60
        self.next_next_map_rotation = None if self.rotate_map_every is None else time.perf_counter() + self.rotate_map_every 

        self.agent_pauses = agent_pauses # for map change
        
        self.sync_mode = sync_mode
        
        self.env_settings_cond = env_settings_cond
        self.step_cond = step_cond
        
        
        

    def _collision_data(self, collision):
        self.collisions.put(collision)


    def _destroy_car_npc(self, car_npc):
        """Destroys a vehicle NPC"""
        if car_npc in self.spawned_car_npcs:
            for actor in self.spawned_car_npcs[car_npc]:
                
                if hasattr(actor, 'set_autopilot'):
                    actor.set_autopilot(False)
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()

                if hasattr(actor, 'is_alive') and actor.is_alive:
                    actor.destroy()
                
            del self.spawned_car_npcs[car_npc]

    def clean_car_npcs(self):
        """Clean up for Map Change"""
        for car_npc in self.spawned_car_npcs.keys():
            for actor in self.spawned_car_npcs[car_npc]:
                if hasattr(actor, 'set_autopilot'):
                    actor.set_autopilot(False)
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()
                if hasattr(actor, 'is_alive') and actor.is_alive:
                    actor.destroy()
        self.spawned_car_npcs = {}

    
    def update_settings_in_loop(self):
        """Main Loop"""
        self.world_name = None

        self.weather = None

        # setup
        while True:
            
            for agent_pause in self.agent_pauses:
                agent_pause.value = 0

            try:

                if self.stop is not None and self.stop.value == STOP.stopping:
                    self.state = CARLA_SETTINGS_STATE.finished
                    return

                if self.restart:
                    self.state = CARLA_SETTINGS_STATE.restarting
                    time.sleep(0.1)
                    continue

                self.clean_car_npcs()

                self.client = carla.Client(*settings.CARLA_HOSTS[self.process_no][:2])
                self.client.set_timeout(5.0)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
                self.world_name = self.map.name

                
                if self.weather is None:
                    self.weather = Weather(self.world.get_weather())
                else:
                    self.weather.set_new_weather(self.world.get_weather())
                if settings.WEATHER_PRESET:
                        weather_preset = find_weather_preset(settings.WEATHER_PRESET)
                        if weather_preset:
                            self.world.set_weather(weather_preset)

                self.car_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
                self.car_blueprints = [x for x in self.car_blueprints if int(x.get_attribute('number_of_wheels')) == 4 and not x.id in settings.DISALLOWED_NPC_VEHICLES]

                self.spawn_points = self.map.get_spawn_points()


                car_despawn_tick = 0

                self.state = CARLA_SETTINGS_STATE.working

            except Exception as e:
                print(e)
                self.state = CARLA_SETTINGS_STATE.error
                time.sleep(1)
                continue
                
            # runs continously
            while True:
                if self.sync_mode:
                    with self.env_settings_cond:
                        self.env_settings_cond.wait()

                step_start = time.perf_counter()

                if self.stop is not None and self.stop.value == STOP.stopping:
                    self.state = CARLA_SETTINGS_STATE.finished
                    return

                # If restart flag is being set, break inner loop
                if self.restart:
                    break

                try:
                    # map rotation
                    if self.next_next_map_rotation and time.perf_counter() > self.next_next_map_rotation:
                       
                        self.state = CARLA_SETTINGS_STATE.restarting
                        self.clean_car_npcs()

                        for agent_pause in self.agent_pauses:
                            agent_pause.value = 1

                        for agent_pause in self.agent_pauses:
                            while agent_pause.value != 2:
                                time.sleep(0.1)

                        available_maps = [map for map in self.client.get_available_maps() if not map.endswith('Opt')]
                        current_map = self.client.get_world().get_map().name
                        map_choice = random.choice(list({map.split('/')[-1] for map in available_maps} - {current_map.split('/')[-1]})) 
                        if settings.MAP_LAYER_GROUND_ONLY:
                            self.client.load_world(map_choice, map_layers=carla.MapLayer.Ground)          
                        else:                 
                            self.client.load_world(map_choice)
                        # needs to be put to sleep to avoid a jenkins/carla error that sometimes occurs, maybe just if the map change speed is set too low. 2bh idk what this is
                        time.sleep(10.0)
                        
                        retries = 0
                        while self.client.get_world().get_map().name.split('/')[-1] != map_choice:
                            retries += 1
                            if retries >= 600:
                                raise Exception('Timeout when waiting for new map to be fully loaded')
                            time.sleep(0.1)
                        
                        for agent_pause in self.agent_pauses:
                            agent_pause.value = 3

                        for agent_pause in self.agent_pauses:
                            retries = 0
                            while agent_pause.value != 0:
                                retries += 1
                                if retries >= 600:
                                    break
                        time.sleep(0.1)
                        self.next_next_map_rotation = time.perf_counter() + self.rotate_map_every
                        break
                        
                    
                    
                    # collision check
                    while not self.collisions.empty():
                        collision = self.collisions.get()

                        car_npc = collision.actor.id
                        self._destroy_car_npc(car_npc)


                    car_despawn_tick += 1
                    
                    # Carla autopilot might cause cars to stop in the middle of intersections blocking whole traffic
                    # Checking for cars stopped at intersections and remove them
                    for car_npc in self.spawned_car_npcs.copy():
                        try:

                            velocity = self.spawned_car_npcs[car_npc][0].get_velocity()
                            simple_speed = velocity.x + velocity.y + velocity.z
                            
                            if simple_speed > 0.1 or simple_speed < -0.1:
                                continue

                            location = self.spawned_car_npcs[car_npc][0].get_location()
                            waypoint = self.map.get_waypoint(location)
                            if not waypoint.is_intersection:
                                continue
                        except:
                            pass
                        self._destroy_car_npc(car_npc)
                        

                    # rotate cars around
                    if self.car_npcs[1] and car_despawn_tick >= self.car_npcs[1] and self.spawned_car_npcs:

                        car_npc = list(self.spawned_car_npcs.keys())[0]
                        self._destroy_car_npc(car_npc)
                        car_despawn_tick = 0
                            
                    # spawn new cars
                    if len(self.spawned_car_npcs) < self.car_npcs[0]:

                        cars_to_spawn = min(10, self.car_npcs[0] - len(self.spawned_car_npcs))

                        retries = 0

                        for _ in range(cars_to_spawn):
                            
                            if retries >= 5:
                                break

                            car_blueprint = random.choice(self.car_blueprints)
                            if car_blueprint.has_attribute('color'):
                                color = random.choice(car_blueprint.get_attribute('color').recommended_values)
                                car_blueprint.set_attribute('color', color)
                            car_blueprint.set_attribute('role_name', 'autopilot')

                            for _ in range(5):
                                try:
                                    spawn_point = random.choice(self.spawn_points)
                                    car_actor = self.world.spawn_actor(car_blueprint, spawn_point)
                                    car_actor.set_autopilot()
                                    break
                                except:
                                    retries += 1
                                    time.sleep(0.001)
                                    continue
                            
                            collision_sensor = self.world.get_blueprint_library().find('sensor.other.collision')
                            colsensor = self.world.spawn_actor(collision_sensor, carla.Transform(), attach_to=car_actor)

                            colsensor.listen(self._collision_data)

                            self.spawned_car_npcs[car_actor.id] = [car_actor, colsensor]
                            
                     
                    if settings.DYNAMIC_WEATHER:
                        self.weather.tick(self.speed_factor)
                        self.world.set_weather(self.weather.weather)   

                    self.stats[0] = len(self.spawned_car_npcs)
                    self.stats[1] = self.weather._sun.azimuth
                    self.stats[2] = self.weather._sun.altitude
                    self.stats[3] = self.weather._storm.clouds
                    self.stats[4] = self.weather._storm.wind
                    self.stats[5] = self.weather._storm.rain

                    self.state = CARLA_SETTINGS_STATE.working


                    sleep_time = self.speed_factor - (time.perf_counter() - step_start)
                    if sleep_time > 0 and not self.sync_mode                                                                                                                                                                                                                                                                                                                                                                                                                                                                        :
                        time.sleep(sleep_time)
                    if self.sync_mode:
                        self.world.tick()
                        with self.step_cond:
                            self.step_cond.notify_all()
                except Exception as e:
                    print(str(e))
                    self.state = CARLA_SETTINGS_STATE.error