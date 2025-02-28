import math
import os
import settings
import pickle
import time
import numpy as np
from Dependencies import CarlaEnv, STOP, models, ACTIONS_NAMES, noise, MPC, MPCToCarla, TensorB, NeptuneLogger
from collections import deque
from threading import Thread
from dataclasses import dataclass
import cv2
import logging
import re
import csv
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from keras.optimizers import Adam
from keras.models import load_model, Model

class ARTDQNAgent:
    def __init__(self, rl_algorithm, model_path=False, id=None):
        """Responsible for prediction and model creation."""
        # for testing, when training we use weights from main model
        self.model_path = model_path

        self.show_conv_cam = (id + 1) in settings.CONV_CAM_AGENTS if id is not None else False

        self.id = id

        self.rl_algorithm = rl_algorithm

        # noise only for ddpg and td3, dqn uses epsilon greedy
        self.noise = getattr(noise, settings.NOISE_TYPE)() if self.rl_algorithm in ['ddpg','td3'] else None
        
        # main model (agent does not use target model)
        self.model = self.create_model(prediction=True)

        # indicator if currently newest weights are the one agent's model is already using
        self.weights_iteration = 0

        self.terminate = False

        self.mpc_exploration_epsilon = settings.MPC_EXPLORATION_START_EPSILON



    def create_model(self, prediction=False, critic_number=None):
        """Load or create a model, used by both agent and trainer."""
        # If there is a path to the model set, load model
        if self.model_path:

            if self.rl_algorithm == 'dqn':
                model_path = self.model_path + '_dqn.model'
                model = load_model(model_path)

            elif self.rl_algorithm == 'ddpg':
                model_path = self.model_path + ('_ddpg_actor.model' if prediction else '_ddpg_critic.model')
                model = load_model(model_path)


            elif self.rl_algorithm == 'td3':
                model_path = self.model_path + ('_td3_actor.model' if prediction else '_td3_critic1.model' if critic_number==1 else '_td3_critic2.model')
                model = load_model(model_path)
                
            
            if prediction:
                self._extract_model_info(model)
                if self.show_conv_cam:
                    model = Model(model.input, [model.output, model.layers[self.convcam_layer].output])

            return model
        

        # else create one
        model_inputs = []
        if "camera" in settings.MODEL_BASES:
            img_color_channels = 1 if settings.FRONT_CAM_TYPE == 'depth' or (settings.FRONT_CAM_TYPE == 'rgb' and settings.RGB_CAM_IMG_TYPE == 'grayscaled') else 3
            
            camera_model = getattr(models, 'model_base_' + settings.MODEL_BASES["camera"])(
                (settings.FRONT_CAM_SETTINGS[0], settings.FRONT_CAM_SETTINGS[1], img_color_channels),
                settings.MODEL_CAMERA_BASE_SETTINGS, base_name="camera"
            )
            model_inputs.append(camera_model)
    
    
        if 'lidar' in settings.MODEL_BASES:
            lidar_input_size = settings.BEV_GRID_SIZE[0], settings.BEV_GRID_SIZE[1], 3 if settings.LIDAR_PREPROCESSING_METHOD == 'birds_eye_view' else 0
            
            lidar_model = getattr(models, 'model_base_' + settings.MODEL_BASES["lidar"])(
                lidar_input_size,
                settings.MODEL_LIDAR_BASE_SETTINGS, base_name="lidar"
            )
            model_inputs.append(lidar_model)

        

        state_encoder = getattr(models, 'state_encoder')(model_inputs=model_inputs, model_settings=settings.MODEL_INPUTS)


        if self.rl_algorithm == 'dqn':
            # DQN model head for discrete actions
            model = getattr(models, 'model_head_dqn')(state_encoder=state_encoder, output_size=len(settings.ACTIONS))
        else:
            # DDPG model heads for continuous actions
            if prediction:
                # If this is for prediction (actor network for DDPG)
                model = getattr(models, 'actor_head_ddpg')(state_encoder=state_encoder, output_size=2) 
                if settings.MODEL_INITIAL_BIAS:
                    model.get_layer('outputs').bias.assign(settings.MODEL_INITIAL_BIAS)
            else:
                # critic network for DDPG (used by the trainer)
                model = getattr(models, 'critic_head_ddpg')(state_encoder=state_encoder, action_dim=2)  
          

        self._extract_model_info(model)

        # we only need to compile for dqn where we use .fit calls, for ddpg/td3 we do it manually with gradient tape
        if not prediction and self.rl_algorithm == 'dqn':
            lr = settings.DQN_OPTIMIZER_LEARNING_RATE
            decay = settings.DQN_OPTIMIZER_DECAY
            self.compile_model(model=model, lr=lr, decay=decay)

        elif self.show_conv_cam:

            model = Model(model.input, [model.output, model.layers[self.convcam_layer].output])

        return model


    def _extract_model_info(self, model):
        """Get conv cam and model name."""
        model_architecture = []
        cnn_kernels = []
        last_conv_layer = None
        for index, layer in enumerate(model.layers):
            layer_name = layer.__class__.__name__.split('_')[-1]
            if layer_name == 'Activation' or layer_name == 'InputLayer':
                if layer_name == 'Activation' and settings.CONV_CAM_LAYER == 'auto_act' and index == last_conv_layer + 1:
                    last_conv_layer += 1
                continue

            if layer_name.startswith('Conv'):
                cnn_kernels.append(str(layer.filters))
                last_conv_layer = index

            if layer_name == 'Dropout':
                layer_name = 'DRopout'
            layer_acronym = ''.join(filter(str.isupper, layer_name.replace('1D', '').replace('2D', '').replace('3D', '')))
            if hasattr(layer, 'filters'):
                layer_acronym += str(layer.filters)
            elif hasattr(layer, 'units'):
                layer_acronym += str(layer.units)

            model_architecture.append(layer_acronym)

        model_architecture = '-'.join(model_architecture)
        cnn_kernels = '-'.join(cnn_kernels)

        # doesn't work if the architecture is too large
        #settings.MODEL_NAME = settings.MODEL_NAME.replace('#MODEL_ARCHITECTURE#', model_architecture)
        #settings.MODEL_NAME = settings.MODEL_NAME.replace('#CNN_KERNELS#', cnn_kernels)
        
        # instead we use the used model parameters
        model_name = ""
        for k, v in settings.MODEL_INPUTS.items():
            if v:
                model_name += k + "_"
        for k, v in settings.MODEL_BASES.items():
            if v:
                model_name += k + "_"
        model_name = model_name[:-1]
        settings.MODEL_NAME = model_name
        
        self.convcam_layer = last_conv_layer if settings.CONV_CAM_LAYER in ['auto', 'auto_act'] else settings.CONV_CAM_LAYER

    def compile_model(self, model, lr, decay):
        if self.rl_algorithm == 'dqn':
            model.compile(loss="mse", optimizer=Adam(learning_rate=lr, decay=decay), metrics=['accuracy'])

    def decode_weights(self, weights):
        return pickle.loads(weights.raw)



    def update_weights(self): 
        model_weights = self.decode_weights(self.weights)
        self.model.set_weights(model_weights)


    def update_weights_in_loop(self):
        """Runs in seperate thread. Monitors if there are new weights being saved by the trainer and updates them."""
        if settings.UPDATE_WEIGHTS_EVERY <= 0:
            return

        while True:

            if self.terminate:
                return

            # if trainer's weights are in a newer revision - save them then update
            if self.trainer_weights_iteration.value >= self.weights_iteration + settings.UPDATE_WEIGHTS_EVERY:
                self.weights_iteration = self.trainer_weights_iteration.value + settings.UPDATE_WEIGHTS_EVERY
                self.update_weights()
  
            else:
                time.sleep(0.001)


    def initial_predict(self):
        # generate input based on the model's expected input shape
        if isinstance(self.model.input_shape, list):  # for models with multiple inputs
            model_input = [tf.ones((1,) + shape[1:]).numpy() for shape in self.model.input_shape]
        else:  # for models with a single input
            model_input = tf.ones((1,) + self.model.input_shape[1:]).numpy()

        self.model.predict(model_input, verbose=0)

  


  
    def get_qs(self, Xs):
        """Prediction for DQN."""
        prediction = self.model.predict(Xs, verbose=0)
        if self.show_conv_cam:
            return [prediction[0][0], prediction[1][0]]
        else:
            return [prediction[0]]


    def get_action(self, Xs):
        """Prediction for DDPG/TD3."""
        prediction = self.model.predict(Xs, verbose=0)
        if settings.SCALE_THROTTLE:
            scaled_throttle = (settings.SCALE_THROTTLE[1] - settings.SCALE_THROTTLE[0]) / 2 * prediction[0][0] + \
                  (settings.SCALE_THROTTLE[1] + settings.SCALE_THROTTLE[0]) / 2
            prediction[0][0] = scaled_throttle

        if settings.SCALE_STEER:
            scaled_steer = (settings.SCALE_STEER[1] - settings.SCALE_STEER[0]) / 2 * prediction[0][1] + \
                  (settings.SCALE_STEER[1] + settings.SCALE_STEER[0]) / 2
            prediction[0][1] = scaled_steer
        if self.show_conv_cam:
            return [prediction[0], prediction[1]]
        else:
            return [prediction[0]]


    def get_model_inputs(self, state):
        Xs = []

        # add batch dimension
        if settings.MODEL_INPUTS['front_camera']:
            front_cam_input = np.array([state['front_cam']]) / 255 
            Xs.append(front_cam_input)

        if settings.MODEL_INPUTS['lidar']:
            lidar_input = np.array([state['lidar']])  
            Xs.append(lidar_input)

        if settings.MODEL_INPUTS['relative_pos']:
            navigation_input = np.array([state['navigation']])  
            Xs.append(navigation_input)

        if settings.MODEL_INPUTS['speed']:
            speed_input = np.round((np.array([[state['speed']]])) / settings.MAX_SPEED, 2)  

            Xs.append(speed_input)

        if settings.MODEL_INPUTS['collision']:
            collision_input = np.array([state['collision']])  
            Xs.append(collision_input)

        if settings.MODEL_INPUTS['lane_invasion']:
            lane_invasion_input = np.array([state['lane_invasion']])  
            Xs.append(lane_invasion_input)  

        if settings.MODEL_INPUTS['distance_to_lane_center']:
            lane_center_input = np.array([state['lane_center']])
            Xs.append(lane_center_input)

        return Xs
    


@dataclass(frozen=True)
class AGENT_STATE:
    starting = 0
    testing = 1
    restarting = 2
    finished = 3
    error = 4
    paused = 5


AGENT_STATE_MESSAGE = {
    0: 'STARTING',
    1: 'TESTING',
    2: 'RESTARING',
    3: 'FINISHED',
    4: 'ERROR',
    5: 'PAUSED',
}




def run(id, carla_instance, stop, pause, episode, rl_algorithm, epsilon, show_preview, weights, weights_iteration, transitions, tensorboard_stats, agent_stats, carla_frametimes, seconds_per_episode, steps_per_episode, sync_mode, synchronizer, env_settings_cond, step_cond, use_n_future_states, put_trans_every):
    """Entry point for agent, responsible for communication with carla and trainer classes."""
    
    # Set GPU used for an agent
    if not settings.USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif settings.AGENT_GPU is not None and type(settings.AGENT_GPU) == int:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.AGENT_GPU)
    elif settings.AGENT_GPU is not None and type(settings.AGENT_GPU) == list:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.AGENT_GPU[id])
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                memory_limit = int(settings.AGENT_MEMORY)  
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
        except Exception as e:
              print(e)
              
              
    # create an agent, set weights from object shared by trainer
    agent = ARTDQNAgent(rl_algorithm, id=id)
    agent.weights = weights
    agent.trainer_weights_iteration = weights_iteration
    agent.update_weights()


    while True:
       
        if stop.value == STOP.stopping:
            agent_stats[0] = AGENT_STATE.finished
            return

        try:
            env = CarlaEnv(carla_instance, sync_mode, env_settings_cond, step_cond, rl_algorithm, seconds_per_episode, steps_per_episode, synchronizer)
            break
        except:
            print("SET ERROR")
            agent_stats[0] = AGENT_STATE.error
            time.sleep(1)
    agent_stats[0] = AGENT_STATE.starting

    env.frametimes = carla_frametimes

    fps_counter = deque(maxlen=60)

    weight_updater = Thread(target=agent.update_weights_in_loop, daemon=True)
    weight_updater.start()

    # predict once on any data to initialize predictions
    agent.initial_predict()

    agent_stats[0] = AGENT_STATE.testing

    
    while stop.value != STOP.stopping:
        
        if stop.value == STOP.carla_simulator_error or stop.value == STOP.restarting_carla_simulator:
            time.sleep(0.1)
            continue

        # pause for world change
        if pause.value == 1:
            pause.value = 2
            agent_stats[0] = AGENT_STATE.paused

        # wait for carla to release pause
        if pause.value == 2:
            time.sleep(0.1)
            continue

        # pause lock released, reconnect and run
        if pause.value == 3:
            pause.value = 0
            agent_stats[0] = AGENT_STATE.starting
            try:
                env.destroy_actors()
            except Exception as e:
                print(type(e))
                pass
            try:
                env = CarlaEnv(carla_instance, sync_mode, env_settings_cond, step_cond, rl_algorithm, seconds_per_episode, steps_per_episode, synchronizer)
                env.frametimes = carla_frametimes
            except:
                print(f"Error creating Carla environment: {e}")
                pass

            time.sleep(1)
            continue

        # restarting episode, setup logging
        episode_reward = 0
        step = 1
        if rl_algorithm == 'dqn':
            predicted_qs = [[] for _ in range(env.action_space_size + 1)]
            predicted_actions = [0 for _ in range(env.action_space_size + 1)]
            random_actions = [0 for _ in range(env.action_space_size + 1)]
        else:
            episode_throttle_values = []
            episode_steering_values = []
            episode_throttle_noise = []
            episode_steering_noise = []
            episode_applied_throttle_values = []
            episode_applied_steering_values = []
            episode_action_magnitudes = []
            episode_action_deltas = []
            episode_noise_magnitudes = []
            last_action = None

        mpc_action_counter = 0    
        action_time = 0
        mpc_action_time = 0
        total_speed = 0
        transition_info = deque(maxlen = use_n_future_states + 1)

        
        try:
            env.destroy_actors()
            current_state, reward, agent_view = env.reset()
            current_state = agent.get_model_inputs(current_state)
            transition_info.append((current_state, None, 0, None))
            if agent.noise:
                agent.noise.reset()
                current_throttle_sigma = agent.noise.throttle_params['sigma']
                current_steering_sigma = agent.noise.steering_params['sigma']
            if settings.USE_MPC:
                mpc = MPC()
                mpc.set_reference_trajectory(env.gnss_sensor.spline_x, env.gnss_sensor.spline_y)
                physics_control = env.vehicle.get_physics_control() 
                low_level_mpc_controller = MPCToCarla()
                low_level_mpc_controller.update_vehicle_info(physics_control)
                current_s_mpc = 0
            mpc_action = None   
        except Exception as e:
            print(str(e))
            print("SET ERROR 2")
            agent_stats[0] = AGENT_STATE.error

            try:
                env.destroy_actors()
            except Exception as e:
                print(f"this is the error: {e}")
                print(type(e))
                pass
            try:
                env = CarlaEnv(carla_instance, sync_mode, env_settings_cond, step_cond, rl_algorithm, seconds_per_episode, steps_per_episode, synchronizer)
                env.frametimes = carla_frametimes
            except:
                print(f"Error creating Carla environment: {e}")
                pass

            time.sleep(1)
            continue

        # update weights if updates on episode restart
        if settings.UPDATE_WEIGHTS_EVERY == 0:
            agent.update_weights()
        agent_stats[0] = AGENT_STATE.testing
        
        done = False
        episode_start = episode_end = time.perf_counter()

        last_processed_cam_update = 0

        conv_min = None
        conv_max = None


        # steps
        while True:
            agent_stats[1] = step
            step_start = time.perf_counter()

            # for async mode only
            if not sync_mode and settings.AGENT_SYNCED_WITH_FRAME:
                wait_for_frame_start = time.perf_counter()
                while True:
                    if env.front_camera.last_cam_update > last_processed_cam_update:
                        last_processed_cam_update = env.front_camera.last_cam_update
                        break
                    if time.perf_counter() > wait_for_frame_start + 1:
                        break
                    time.sleep(0.001)
                    
            action_start = time.perf_counter()
            

            # receive action based on action type
            if rl_algorithm == 'dqn':
                # get new action - either random one or predicted by the model
                if np.random.random() > epsilon[0]:
                    qs = agent.get_qs(current_state)
                    predicted_action = np.argmax(qs[0])

                    
                    for i in range(env.action_space_size):
                        predicted_qs[0].append(qs[0][i])
                        predicted_qs[i + 1].append(qs[0][i])
                    predicted_actions[0] += 1
                    predicted_actions[predicted_action + 1] += 1
                    
                else:
                    predicted_action = np.random.randint(0, env.action_space_size)
                    random_actions[0] += 1
                    random_actions[predicted_action + 1] += 1

                if agent.show_conv_cam:
                    conv_output = qs[1]
            ## DDPG or TD3 are the same for exploration
            else:
                model_output = agent.get_action(current_state)
                predicted_action = model_output[0]
                conv_output = model_output[1] if agent.show_conv_cam else None

            action_end = time.perf_counter()
            action_time += action_end - action_start  
            
            
            # compute mpc action
            if settings.USE_MPC:
                mpc.setup_MPC()
                current_mpc_state = env.calculate_mpc_state(current_s_mpc)
                current_s_mpc = current_mpc_state[0]
                try:
                    result = mpc.solve(current_mpc_state)
                    if result is None:
                        raise ValueError("MPC returned None")
                    mpc_control_input, predicted_trajectory = result

                    # high level actions
                    delta, a = mpc_control_input
                    delta = -delta

                    # setup command for translation to caröa actions
                    mpc_command = {
                    "steering_angle": delta,
                    "speed": 25 / 3.6,
                    "acceleration": a,
                    "jerk": 0.0
                }
                    
                    vehicle_status = env.get_vehicle_state()
                    low_level_mpc_controller.update_vehicle_status(vehicle_status)
                    low_level_mpc_controller.update_current_values(current_mpc_state[3])
                    low_level_mpc_controller.set_target_values(mpc_command)

                    low_level_mpc_controller.vehicle_control_cycle()
                    control_output = low_level_mpc_controller.get_output()

                    # Low Level Actions
                    throttle = control_output['throttle'] if control_output['throttle'] > 0 else -control_output['brake']
                    steer = control_output['steer']
                    mpc_action = np.array([throttle, steer])

                    last_action = mpc_action
                except Exception as e:
                    #print(f"MPC found no solution: {e}")
                        done = True
                        break
                                
                
            mpc_end = time.perf_counter()
            mpc_action_time += mpc_end - action_end

            # chose action to be executed
            if settings.USE_MPC and (settings.MPC_PHASE == 'Imitation' or (settings.MPC_PHASE == 'Transition' and np.random.random() < agent.mpc_exploration_epsilon)):
                action = mpc_action
                mpc_action_counter += 1
            else:
                action = predicted_action
            
            ## add noise to action
            if agent.noise:
                noise = agent.noise.sample()

                episode_throttle_values.append(action[0])
                episode_steering_values.append(action[1])
                episode_throttle_noise.append(noise[0])
                episode_steering_noise.append(noise[1])

                
                action[0] = np.clip(action[0] + noise[0], *settings.SCALE_THROTTLE)
                action[1] = np.clip(action[1] + noise[1], *settings.SCALE_STEER)

                episode_applied_throttle_values.append(action[0])
                episode_applied_steering_values.append(action[1])

                action_magnitude = np.linalg.norm(action)
                episode_action_magnitudes.append(action_magnitude)
                noise_magnitude = np.linalg.norm(noise)
                episode_noise_magnitudes.append(noise_magnitude)

                if last_action is not None:
                    action_delta = np.linalg.norm(action - last_action)
                    episode_action_deltas.append(action_delta)

                last_action = action
            
            ## conv cam 
            if agent.show_conv_cam:

                # stabilizes image flickering
                conv_min = np.min(conv_output) if conv_min is None else 0.8 * conv_min + 0.2 * np.min(conv_output)
                conv_max = np.max(conv_output) if conv_max is None else 0.8 * conv_max + 0.2 * np.max(conv_output)

                # normalization
                conv_preview = ((conv_output - conv_min) * 255 / (conv_max - conv_min)).astype(np.uint8)

                # swap axes and reshape to format output image
                conv_preview = np.moveaxis(conv_preview, 1, 2)
                conv_preview = conv_preview.reshape((conv_preview.shape[0], conv_preview.shape[1] * conv_preview.shape[2]))

                # find where to "wrap" wide image
                i = 1
                while not (conv_preview.shape[1] / conv_output.shape[1]) % (i * i):
                    i *= 2
                i //= 2

                # wrap image
                conv_preview_reorganized = np.zeros((conv_preview.shape[0] * i, conv_preview.shape[1] // i), dtype=np.uint8)
                for start in range(i):
                    conv_preview_reorganized[start * conv_preview.shape[0]:(start + 1) * conv_preview.shape[0], 0:conv_preview.shape[1] // i] = conv_preview[:, (conv_preview.shape[1] // i) * start:(conv_preview.shape[1] // i) * (start + 1)]

                cv2.imshow(f'Agent {id + 1} - Convcam', conv_preview_reorganized)
                cv2.waitKey(1)


            
            try:
                # step in environment
                current_state, reward, done, agent_view, additional_data = env.step(action)
                current_state = agent.get_model_inputs(current_state)
            except Exception as e:
                print(e)
                print("SET ERROR 3")
                agent_stats[0] = AGENT_STATE.error
                time.sleep(1)
                break
            
            if show_preview[0] == 1 or show_preview[0] == 2:
                cv2.imshow(f'Agent {id+1} - preview', agent_view)
                cv2.waitKey(1)
                env.preview_camera_enabled = False

            if show_preview[0] >= 10 or show_preview[0] == 3:
                if show_preview[0] == 3:
                    env.preview_camera_enabled = show_preview[1:]
                else:
                    env.preview_camera_enabled = settings.PREVIEW_CAMERA_PRESETS[int(show_preview[0]) - 10]
                if env.preview_camera.image is not None:
                    cv2.imshow(f'Agent {id+1} - preview', env.preview_camera.image)
                    cv2.waitKey(1)

            try:
                if not show_preview[0] and cv2.getWindowProperty(f'Agent {id+1} - preview', 0) >= 0:
                    cv2.destroyWindow(f'Agent {id + 1} - preview')
                    env.preview_camera_enabled = False
            except:
                pass
            
            
            episode_reward += reward
            
            
            # add transition to a queue
            transition_info.append((current_state, action, reward, mpc_action))
            if (step % put_trans_every == 0 or done) and len(transition_info) == transition_info.maxlen:

                states = [item[0] for item in transition_info]
                action = transition_info[1][1]
                reward = sum(item[2] for item in list(transition_info)[1:])
                mpc_action = transition_info[1][3]
                
                # send transition to trainer
                transitions.put_nowait((states, action, reward, 1 if done else 0, mpc_action))

            step += 1
            
            if done:
                episode_end = time.perf_counter()
                
                # decay mpc exploration epsilon
                if settings.USE_MPC and settings.MPC_PHASE == 'Transition':
                    agent.mpc_exploration_epsilon = max(agent.mpc_exploration_epsilon * settings.MPC_EXPLORATION_EPSILON_DECAY, settings.MPC_EXPLORATION_MIN_EPSILON)
                break

       

            

            frame_time = time.perf_counter() - step_start
            fps_counter.append(frame_time)
            agent_stats[2] = len(fps_counter)/sum(fps_counter)
            

        try:
            env.destroy_actors()
        except:
            pass

        if done:
            if step == 0:
                continue
            try:
                # logging
                episode_time = max(episode_end - episode_start, 0.1)
                step_time = episode_time / step
                action_step_time = action_time / step
                mpc_step_time = mpc_action_time / step
                avg_fps = step / episode_time

                mpc_exploration_epsilon = agent.mpc_exploration_epsilon
                avg_pred_time = action_time / step
                total_speed = sum(env.gnss_sensor.speeds)
                avg_speed = total_speed / step
                reward_factor = settings.EPISODE_FPS / avg_fps

                episode_reward_weighted = (episode_reward - reward) * reward_factor + reward
                episode_reward_avg = episode_reward / max(step, 1)
                waypoints_reached = 0
                waypoints_skipped = 0
                cur_sp = env.cur_sp
                if env.gnss_sensor:
                    waypoints_skipped = env.gnss_sensor.skipped_waypoints
                    waypoints_reached = env.gnss_sensor.waypoint_idx - waypoints_skipped - 1
                    waypoints_total = env.gnss_sensor.waypoint_idx - 2
                    waypoints_total_per = (env.gnss_sensor.waypoint_idx) / (len(env.gnss_sensor.route) -1)

                    avg_lateral_error = sum(env.gnss_sensor.lateral_errors) / step
                    avg_yaw_error = sum(env.gnss_sensor.yaw_differences) / step
                per_step_rewards = {}
                episode_end_reason = additional_data['episode_end_reason'] if additional_data else -1
                per_step_rewards['per_step_navigation_rewards'] = additional_data.get('navigation_rewards', 0) 
                per_step_rewards['per_step_lane_center_rewards'] = additional_data.get('lane_center_rewards', 0) 
                per_step_rewards['per_step_speed_rewards'] = additional_data.get('speed_rewards', 0)
                per_step_rewards['per_step_collision_rewards'] = additional_data.get('collision_rewards', 0)
                

                # send logging information to trainer
                if rl_algorithm == 'dqn':
                    avg_predicted_qs = []
                    for i in range(env.action_space_size + 1):
                        if len(predicted_qs[i]):
                            avg_predicted_qs.append(sum(predicted_qs[i])/len(predicted_qs[i]))
                            avg_predicted_qs.append(np.std(predicted_qs[i]))
                            avg_predicted_qs.append(100 * predicted_actions[i] / predicted_actions[0])
                        else:
                            avg_predicted_qs.append(-10**6)
                            avg_predicted_qs.append(-10**6)
                            avg_predicted_qs.append(-10**6)
            
                    with episode.get_lock():
                        episode.value += 1
                        tensorboard_stats.put([episode.value, episode_reward, episode_reward_avg, epsilon[0], episode_time, agent_stats[2], episode_reward_weighted, predicted_actions[0], random_actions[0], avg_pred_time, avg_speed, waypoints_reached, waypoints_skipped, waypoints_total, waypoints_total_per, episode_end_reason, step, per_step_rewards, step_time, action_step_time, mpc_step_time, cur_sp, avg_lateral_error, avg_yaw_error, mpc_exploration_epsilon, mpc_action_counter] + avg_predicted_qs)

                    # decay epsilon for dqn
                    # epsilon is an array of 3 elements: current epsilon [0], epsilon decay value [1] and minimal epsilon to heep [2]
                    if epsilon[0] > epsilon[2]:
                        with epsilon.get_lock():
                            epsilon[0] *= epsilon[1]
                            epsilon[0] = max(epsilon[2], epsilon[0])
                    
                
                else:
                    with episode.get_lock():
                        episode.value += 1
                        tensorboard_stats.put([episode.value, episode_reward, episode_reward_avg, episode_time, agent_stats[2], episode_reward_weighted, avg_pred_time, avg_speed, waypoints_reached, waypoints_skipped, waypoints_total, waypoints_total_per, episode_end_reason, step, per_step_rewards, step_time, action_step_time, mpc_step_time, cur_sp, avg_lateral_error, avg_yaw_error, mpc_exploration_epsilon, mpc_action_counter, episode_throttle_values, episode_steering_values, episode_throttle_noise, episode_steering_noise, episode_applied_throttle_values, episode_applied_steering_values, episode_action_magnitudes, episode_action_deltas, episode_noise_magnitudes, current_throttle_sigma, current_steering_sigma])
            except:
                ...

        agent_stats[0] = AGENT_STATE.restarting
        agent_stats[1] = 0
        agent_stats[2] = 0


    agent.terminate = True
    weight_updater.join()

    agent_stats[0] = AGENT_STATE.finished

    transitions.cancel_join_thread()
    tensorboard_stats.cancel_join_thread()
    carla_frametimes.cancel_join_thread()



def test(model_path, sync_mode, env_settings_cond, step_cond, pause, synchronizer, console_print_callback):
    """Run algorithm through test scenarios."""
    rl_algorithm = settings.TESTING_RL_ALGORITHM
    show_preview = settings.TESTING_PREVIEW 
    
    # setup logging
    logdir = "logs/{}-{}".format(model_path, int(time.time()))
    tensorboard = TensorB(log_dir=logdir)
    tensorboard.step = settings.TESTING_ROUTE_AREA[0]
    if 'neptune' in settings.ADDITIONAL_LOGGING:
        neptune_logger = NeptuneLogger() if 'neptune' in settings.ADDITIONAL_LOGGING else None
        neptune_logger.step = settings.TESTING_ROUTE_AREA if 'neptune' in settings.ADDITIONAL_LOGGING else None


    # log csv file
    folder_name = "test_logs"
    os.makedirs(folder_name, exist_ok=True)
    existing_files = os.listdir(folder_name)
    log_files = [f for f in existing_files if f.startswith("test_log_") and f.endswith(".csv")]

    # get highest int for # of log file
    log_numbers = [int(re.search(r'(\d+)', f).group(0)) for f in log_files if re.search(r'(\d+)', f)]
    next_log_number = max(log_numbers, default=0) + 1

    log_file_name = f"test_log_{next_log_number}.csv"
    log_file_path = os.path.join(folder_name, log_file_name)

    
    header = [
    'route', 'completed', 'completion_percentage', 'steps', 'episode_total_time', 
    'per_step_time', 'waypoints_reached', 'waypoints_skipped', 
    'total_waypoints', 'min_speed_mps', 'max_speed_mps', 
    'avg_speed_mps', 'rmse_speed_mps', 'min_lateral_error_m', 
    'max_lateral_error_m', 'avg_lateral_error_m', 'rmse_lateral_error_m', 
    'min_heading_error_rad', 'max_heading_error_rad', 
    'avg_heading_error_rad', 'rmse_heading_error_rad'
]

    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

    # create agent and load model
    agent = ARTDQNAgent(rl_algorithm, model_path, id=0)
    agent.initial_predict()
    env = CarlaEnv(0, sync_mode, env_settings_cond, step_cond, rl_algorithm, None, settings.TESTING_MAX_STEPS, synchronizer, testing=True)
    env.frametimes = deque(maxlen=60)

    env.preview_camera_enabled = settings.PREVIEW_CAMERA_PRESETS[0]


    # main loop
    for sp in range(*settings.TESTING_ROUTE_AREA):
        # restarting episode
        step = 1
        episode_throttle_values = []
        episode_steering_values = []
        
        try:
            env.destroy_actors()
            current_state, _, _ = env.reset(sp)
            current_state = agent.get_model_inputs(current_state)
            # no model path means we want mpc actions
            if not model_path:
                mpc = MPC()
                mpc.set_reference_trajectory(env.gnss_sensor.spline_x, env.gnss_sensor.spline_y)
                physics_control = env.vehicle.get_physics_control() 
                low_level_mpc_controller = MPCToCarla()
                low_level_mpc_controller.update_vehicle_info(physics_control)
                current_s_mpc = 0
        except Exception as e:
            print(str(e))
        
        done = False
        episode_start = time.perf_counter()
        # loop over steps
        while True:
            
            # if a model is provided we use it, else it defaults to use mpc actions
            if model_path:
                if rl_algorithm == 'dqn':
                    qs = agent.get_qs(current_state)
                    action = np.argmax(qs[0])
                
                ## DDPG or TD3
                else:
                    model_output = agent.get_action(current_state)
                    action = model_output[0]

            else:
                mpc.setup_MPC()
                current_mpc_state = env.calculate_mpc_state(current_s_mpc)
                current_s_mpc = current_mpc_state[0]

                try:
                    result = mpc.solve(current_mpc_state)
                    if result is None:
                        raise ValueError("MPC returned None")
                    mpc_control_input, predicted_trajectory = result

                    delta, a = mpc_control_input
                    delta = -delta

                    mpc_command = {
                    "steering_angle": delta,
                    "speed": 25 / 3.6,
                    "acceleration": a,
                    "jerk": 0.0
                }
                    
                    vehicle_status = env.get_vehicle_state()
                    low_level_mpc_controller.update_vehicle_status(vehicle_status)
                    low_level_mpc_controller.update_current_values(current_mpc_state[3])
                    low_level_mpc_controller.set_target_values(mpc_command)

                    low_level_mpc_controller.vehicle_control_cycle()
                    control_output = low_level_mpc_controller.get_output()

                    throttle = control_output['throttle'] if control_output['throttle'] > 0 else -control_output['brake']
                    steer = control_output['steer']
                    action = np.array([throttle, steer])
                except Exception as e:
                        print("MPC NO SOLUTION")
                        done = True
                        break
                
            episode_throttle_values.append(action[0])
            episode_steering_values.append(action[1])
            current_state, _, done, _, _ = env.step(action)
            current_state = agent.get_model_inputs(current_state)
            

            if show_preview:
                env.preview_camera_enabled = settings.TESTING_CAMERA
                if env.preview_camera.image is not None:
                    cv2.imshow(f'Agent {1} - preview', env.preview_camera.image)
                    cv2.waitKey(1)


            step += 1
            #console_print_callback(fps_counter, env, qs[0], action, ACTIONS_NAMES[env.actions[action]])
            if step >= settings.TESTING_MAX_STEPS:
                done = True
            if done:
                episode_end = time.perf_counter()
                break
        
        env.destroy_actors()
        if done:

            # logging
            episode_time = episode_end - episode_start
            step_time = episode_time / step
            waypoints_reached = 0
            waypoints_skipped = 0
            cur_sp = env.cur_sp
            if env.gnss_sensor:
                waypoints_skipped = env.gnss_sensor.skipped_waypoints
                waypoints_reached = env.gnss_sensor.waypoint_idx - waypoints_skipped - 1
                waypoints_total = env.gnss_sensor.waypoint_idx - 2
                waypoints_total_per = (env.gnss_sensor.waypoint_idx) / (len(env.gnss_sensor.route) -1)
                speeds = env.gnss_sensor.speeds
                lateral_errors = env.gnss_sensor.lateral_errors
                heading_errors = env.gnss_sensor.yaw_differences


            min_len_actions = min(len(episode_throttle_values), len(episode_steering_values))
            actions = pd.DataFrame({
                "timestep": list(range(min_len_actions)),
                "throttle": episode_throttle_values[:min_len_actions],
                "steering": episode_steering_values[:min_len_actions],
            })

            min_len_values = min(len(speeds), len(lateral_errors), len(heading_errors))
            values = pd.DataFrame({
                "timestep": list(range(min_len_values)),
                "speed": speeds[:min_len_values],
                "lateral_error": lateral_errors[:min_len_values],
                "heading_error": heading_errors[:min_len_values]
            })

            speeds = np.array(env.gnss_sensor.speeds)
            lateral_errors = np.array(env.gnss_sensor.lateral_errors)
            heading_errors = np.array(env.gnss_sensor.yaw_differences)

            min_speed = np.min(speeds)
            max_speed = np.max(speeds)
            avg_speed = np.mean(speeds)
            rmse_speed = np.sqrt(np.mean(speeds ** 2))

            min_lateral_error = np.min(lateral_errors)
            max_lateral_error = np.max(lateral_errors)
            avg_lateral_error = np.mean(lateral_errors)
            rmse_lateral_error = np.sqrt(np.mean(lateral_errors ** 2))

            min_heading_error = np.min(heading_errors)
            max_heading_error = np.max(heading_errors)
            avg_heading_error = np.mean(heading_errors)
            rmse_heading_error = np.sqrt(np.mean(heading_errors ** 2))

            stats = {
            'route': sp,
            'completed': True if waypoints_total_per == 1 else False,
            'completion_percentage': waypoints_total_per,
            'steps': step,
            'episode_total_time': episode_time,
            'per_step_time': step_time,
            'waypoints_reached': waypoints_reached,
            'waypoints_skipped': waypoints_skipped,
            'total_waypoints': waypoints_total,

            'min_speed_mps': min_speed,
            'max_speed_mps': max_speed,
            'avg_speed_mps': avg_speed,
            'rmse_speed_mps': rmse_speed,

            'min_lateral_error_m': min_lateral_error,
            'max_lateral_error_m': max_lateral_error,
            'avg_lateral_error_m': avg_lateral_error,
            'rmse_lateral_error_m': rmse_lateral_error,

            'min_heading_error_rad': min_heading_error,
            'max_heading_error_rad': max_heading_error,
            'avg_heading_error_rad': avg_heading_error,
            'rmse_heading_error_rad': rmse_heading_error,
            }
            
            tensorboard.update_stats("test", sp, **stats)
            if 'neptune' in settings.ADDITIONAL_LOGGING:
                neptune_logger.update_stats("test", sp ,**stats)
                neptune_logger.log_df('test/actions', f'actions_{sp}', actions)
                neptune_logger.log_df('test/values', f'values_{sp}', values)
                
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writerow(stats)