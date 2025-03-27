from abc import ABC, abstractmethod
import math
import settings
import carla
import numpy as np
import time
from dataclasses import dataclass
from scipy.interpolate import CubicSpline


'''
Implementation of all Sensors. Every Sensor needs a callback and a destroy method.
'''

class SensorInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def callback(self, event):
        pass

    @abstractmethod
    def destroy(self):
        pass
    
    
class RGB_camera(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings, frametimes=None, testing=False):
        self.image = None
        self.img_width = sensor_settings[0]
        self.img_height = sensor_settings[1]
        self.frametimes = frametimes
        self.testing = testing
        self.img_type = sensor_settings[6]
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', f'{self.img_width}')
        rgb_bp.set_attribute('image_size_y', f'{self.img_height}')
        rgb_bp.set_attribute('fov', f'{sensor_settings[5]}')
        rgb_bp.set_attribute('bloom_intensity', f'{sensor_settings[7]}')
        rgb_bp.set_attribute('fstop', f'{sensor_settings[8]}')
        rgb_bp.set_attribute('iso', f'{sensor_settings[9]}')
        rgb_bp.set_attribute('gamma', f'{sensor_settings[10]}')
        rgb_bp.set_attribute('lens_flare_intensity', f'{sensor_settings[11]}')
        rgb_bp.set_attribute('sensor_tick', f'{sensor_settings[12]}')
        rgb_bp.set_attribute('shutter_speed', f'{sensor_settings[13]}')
        rgb_bp.set_attribute('lens_circle_falloff', f'{sensor_settings[14]}')
        rgb_bp.set_attribute('lens_circle_multiplier', f'{sensor_settings[15]}')
        rgb_bp.set_attribute('lens_k', f'{sensor_settings[16]}')
        rgb_bp.set_attribute('lens_kcube', f'{sensor_settings[17]}')
        rgb_bp.set_attribute('lens_x_size', f'{sensor_settings[18]}')
        rgb_bp.set_attribute('lens_y_size', f'{sensor_settings[19]}')
        transform = carla.Transform(carla.Location(x=sensor_settings[2], y=sensor_settings[3], z=sensor_settings[4]))
                
        self.sensor = world.spawn_actor(rgb_bp, transform, attach_to=vehicle)
        self.sensor.listen(self.callback)
        self.last_cam_update = time.time()

    def callback(self, image):
        image = np.array(image.raw_data)
        image = image.reshape((self.img_height, self.img_width, 4))
        image = image[:, :, :3]

        if self.img_type == 'rgb':
            self.image = image
        elif self.img_type == 'greyscaled':
            self.image = np.expand_dims(np.dot(image, [0.299, 0.587, 0.114]).astype('uint8'), -1)
        elif self.img_type == 'stacked':
            if self.image is None:
                image = np.dot(image, [0.299, 0.587, 0.114]).astype('uint8')
                self.image = np.stack([image, image, image], axis=-1)
            else:
                self.image = np.roll(self.image, 1, -1)
                self.image[..., 0] = np.dot(image, [0.299, 0.587, 0.114]).astype('uint8')

        if self.frametimes is not None:
            self.last_cam_update = time.time()
            if self.testing:
                self.frametimes.append(time.time() - self.last_cam_update)
            else:
                self.frametimes.put_nowait(time.time() - self.last_cam_update)
            

    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'destroy'):
            self.sensor.destroy()
        


class Preview_camera(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings, on=True):
        self.image = None
        self.on = on
        self.img_width = int(sensor_settings[0])
        self.img_height = int(sensor_settings[1])
        
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('fov', f'{sensor_settings[5]}')
        rgb_bp.set_attribute('image_size_x', f'{self.img_width}')
        rgb_bp.set_attribute('image_size_y', f'{self.img_height}')
        transform = carla.Transform(carla.Location(x=sensor_settings[2], y=sensor_settings[3], z=sensor_settings[4]))
        
        self.sensor = world.spawn_actor(rgb_bp, transform, attach_to=vehicle)
        self.sensor.listen(self.callback)


    def callback(self, image):

        image = np.array(image.raw_data)
        image = image.reshape((self.img_height, self.img_width, 4))
        image = image[:, :, :3]

        self.image = image


    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()



class Collision_detector(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle):
        self.collision_hist = []

        col_bp = blueprint_library.find("sensor.other.collision")
        self.sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self.callback)

    def callback(self, event):      
        collision_actor_id = event.other_actor.type_id
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        for actor_id, impulse in settings.COLLISION_FILTER:
            if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
                return

        self.collision_hist.append(event)

    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()



class Depth_camera(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings, frametimes=None, testing=False):
        self.image = None
        self.img_width = sensor_settings[0]
        self.img_height = sensor_settings[1]
        self.frametimes = frametimes
        self.testing = testing

        depth_bp = blueprint_library.find('sensor.camera.rgb')
        depth_bp.set_attribute('image_size_x', f'{self.img_width}')
        depth_bp.set_attribute('image_size_y', f'{self.img_height}')
        depth_bp.set_attribute('fov', f'{sensor_settings[5]}')
        depth_bp.set_attribute('sensor_tick', f'{sensor_settings[6]}')
        depth_bp.set_attribute('lens_circle_falloff', f'{sensor_settings[7]}')
        depth_bp.set_attribute('lens_circle_multiplier', f'{sensor_settings[8]}')
        depth_bp.set_attribute('lens_k', f'{sensor_settings[9]}')
        depth_bp.set_attribute('lens_kcube', f'{sensor_settings[10]}')
        depth_bp.set_attribute('lens_x_size', f'{sensor_settings[11]}')
        depth_bp.set_attribute('lens_y_size', f'{sensor_settings[12]}')

        transform = carla.Transform(carla.Location(x=sensor_settings[2], y=sensor_settings[3], z=sensor_settings[4]))


        self.sensor = world.spawn_actor(depth_bp, transform, attach_to=vehicle)
        self.sensor.listen(self.callback)

    def callback(self, image):
        image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image = np.reshape(image, (self.img_height, self.img_width, 4))
        image = image[:, :, :3]

        self.image = image

        if self.frametimes is not None:
            self.last_cam_update = time.time()
            if self.testing:
                self.frametimes.append(time.time() - self.last_cam_update)
            else:
                self.frametimes.put_nowait(time.time() - self.last_cam_update)


    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()


class IMU_Sensor(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings):
        self.event = None
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', f'{sensor_settings[3]}')
        imu_bp.set_attribute('noise_accel_stddev_x', str(sensor_settings[4]))
        imu_bp.set_attribute('noise_accel_stddev_y', str(sensor_settings[5]))
        imu_bp.set_attribute('noise_accel_stddev_z', str(sensor_settings[6]))
        imu_bp.set_attribute('noise_gyro_bias_x', str(sensor_settings[7]))
        imu_bp.set_attribute('noise_gyro_bias_y', str(sensor_settings[8]))
        imu_bp.set_attribute('noise_gyro_bias_z', str(sensor_settings[9]))
        imu_bp.set_attribute('noise_gyro_stddev_x', str(sensor_settings[10]))
        imu_bp.set_attribute('noise_gyro_stddev_y', str(sensor_settings[11]))
        imu_bp.set_attribute('noise_gyro_stddev_z', str(sensor_settings[12]))
        imu_bp.set_attribute('noise_seed', str(sensor_settings[13]))
        transform = carla.Transform(carla.Location(x=sensor_settings[0], y=sensor_settings[1], z=sensor_settings[2]))

        self.sensor = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
        self.sensor.listen(self.callback)

        self.last_steering_value = None
    def callback(self, event):
        self.event = event




    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()



@dataclass(frozen=True)
class LANE_MARKING_TYPES:
    NONE: int = carla.LaneMarkingType.NONE
    SOLID: int = carla.LaneMarkingType.Solid
    BROKEN: int = carla.LaneMarkingType.Broken
    SOLID_SOLID: int = carla.LaneMarkingType.SolidSolid
    SOLID_BROKEN: int = carla.LaneMarkingType.SolidBroken
    BROKEN_SOLID: int = carla.LaneMarkingType.BrokenSolid
    BROKEN_BROKEN: int = carla.LaneMarkingType.BrokenBroken
    BOTTS_DOTS: int = carla.LaneMarkingType.BottsDots
    CURB: int = carla.LaneMarkingType.Curb
    GRASS: int = carla.LaneMarkingType.Grass

    @property
    def mapping(self):
        return {
            self.NONE: "NONE",
            self.SOLID: "SOLID",
            self.BROKEN: "BROKEN",
            self.SOLID_SOLID: "SOLID_SOLID",
            self.SOLID_BROKEN: "SOLID_BROKEN",
            self.BROKEN_SOLID: "BROKEN_SOLID",
            self.BROKEN_BROKEN: "BROKEN_BROKEN",
            self.BOTTS_DOTS: "BOTTS_DOTS",
            self.CURB: "CURB",
            self.GRASS: "GRASS"
        }



class Lane_invasion_detector(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings):
        self.data = []
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        transform = carla.Transform(carla.Location(x=sensor_settings[0], y=sensor_settings[1], z=sensor_settings[2]))


        self.lane_marking_types = LANE_MARKING_TYPES()
        self.sensor = world.spawn_actor(lane_invasion_bp, transform, attach_to=vehicle)
        self.sensor.listen(self.callback)

    def callback(self, event):
        lane_markings = event.crossed_lane_markings
        self.data = []
        for marking in lane_markings:
            marking_type = self.lane_marking_types.mapping.get(marking.type, "UNKNOWN")
            #for filter_marking, penalty in settings.LANE_INVASION_FILTER:
            #    if marking_type == filter_marking:
            #        self.lane_invasion_hist.append(penalty)
            self.data.append(marking_type)

    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()


class Lidar(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings):

        self.data = None
        self.preprocessing_method = sensor_settings[3]
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        
        lidar_bp.set_attribute('channels', f'{sensor_settings[4]}') 
        lidar_bp.set_attribute('range', f'{sensor_settings[5]}')  
        lidar_bp.set_attribute('points_per_second', f'{sensor_settings[6]}') 
        lidar_bp.set_attribute('rotation_frequency', f'{sensor_settings[7]}') 
        lidar_bp.set_attribute('upper_fov', f'{sensor_settings[8]}') 
        lidar_bp.set_attribute('lower_fov', f'{sensor_settings[9]}') 
        lidar_bp.set_attribute('horizontal_fov', f'{sensor_settings[10]}')  
        lidar_bp.set_attribute('atmosphere_attenuation_rate', f'{sensor_settings[11]}')  
        lidar_bp.set_attribute('dropoff_general_rate', f'{sensor_settings[12]}')
        lidar_bp.set_attribute('dropoff_intensity_limit', f'{sensor_settings[13]}') 
        lidar_bp.set_attribute('dropoff_zero_intensity', f'{sensor_settings[14]}')  
        lidar_bp.set_attribute('sensor_tick', f'{sensor_settings[15]}')  
        
        transform = carla.Transform(carla.Location(x=sensor_settings[0], y=sensor_settings[1], z=sensor_settings[2]))
        self.sensor = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
        
        self.sensor.listen(self.callback)
        

    def callback(self, data):
        point_cloud = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1,4)
        if self.preprocessing_method == 'point_cloud':
            self.data = point_cloud
            
        if self.preprocessing_method == 'birds_eye_view':
            self.point_cloud_to_bev_image(point_cloud)
        if self.preprocessing_method == 'voxel_grid':
            pass

        if self.preprocessing_method == '...':
            pass

    def point_cloud_to_bev_image(self, point_cloud):
        point_cloud = point_cloud[(point_cloud[:, 2] > settings.BEV_Z_THRESHOLD[0]) & (point_cloud[:, 2] < settings.BEV_Z_THRESHOLD[1])]

        bev_image = np.zeros((settings.BEV_GRID_SIZE[0], settings.BEV_GRID_SIZE[1], 3), dtype=np.float32)  # 3 channels: height, intensity, density

        # Calculate grid indices
        x_indices = ((point_cloud[:, 0] + (settings.BEV_GRID_SIZE[1] * settings.BEV_GRID_RESOLUTION) / 2) / settings.BEV_GRID_RESOLUTION).astype(np.int32)
        y_indices = ((point_cloud[:, 1] + (settings.BEV_GRID_SIZE[0] * settings.BEV_GRID_RESOLUTION) / 2) / settings.BEV_GRID_RESOLUTION).astype(np.int32)


        if settings.BEV_CLAMP_POINTS:
            # Clamp indices
            x_indices = np.clip(x_indices, 0, settings.BEV_GRID_SIZE[1] - 1)
            y_indices = np.clip(y_indices, 0, settings.BEV_GRID_SIZE[0] - 1)
        
        else:
            # filter out-of-bound indices
            mask = (
                (x_indices >= 0) & (x_indices < settings.BEV_GRID_SIZE[1]) &
                (y_indices >= 0) & (y_indices < settings.BEV_GRID_SIZE[0])
            )

            x_indices = x_indices[mask]
            y_indices = y_indices[mask]
            point_cloud = point_cloud[mask]


            # Fill the BEV image
        for i in range(len(point_cloud)):
            x, y, z, intensity = x_indices[i], y_indices[i], point_cloud[i, 2], point_cloud[i, 3]

            bev_image[y, x, 0] = max(bev_image[y, x, 0], z)

            bev_image[y, x, 1] = max(bev_image[y, x, 1], intensity)

            bev_image[y, x, 2] += 1

        # Normalization
        bev_image[:, :, 2] = np.log1p(bev_image[:, :, 2])  

        if bev_image[:, :, 0].max() != bev_image[:, :, 0].min(): 
            bev_image[:, :, 0] = (bev_image[:, :, 0] - bev_image[:, :, 0].min()) / (bev_image[:, :, 0].max() - bev_image[:, :, 0].min())
        
        if bev_image[:, :, 1].max() != bev_image[:, :, 1].min():  
            bev_image[:, :, 1] = (bev_image[:, :, 1] - bev_image[:, :, 1].min()) / (bev_image[:, :, 1].max() - bev_image[:, :, 1].min())

        self.data = bev_image

    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()


class Radar(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings):
        self.radar_data = []
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', f'{sensor_settings[3]}')
        radar_bp.set_attribute('vertical_fov', f'{sensor_settings[4]}')
        radar_bp.set_attribute('range', f'{sensor_settings[5]}')
        radar_bp.set_attribute('points_per_second', f'{sensor_settings[6]}')
        radar_bp.set_attribute('sensor_tick', f'{sensor_settings[7]}')
        transform = carla.Transform(carla.Location(x=sensor_settings[0],y=sensor_settings[1], z=sensor_settings[2]))
        
        self.sensor = world.spawn_actor(radar_bp, transform, attach_to=vehicle)
        self.sensor.listen(self.callback)

    def callback(self, data):
        detections = []
        for detection in data:
            feature_vector = [
                detection.altitude,  
                detection.azimuth,  
                detection.depth,     
                detection.velocity   
            ]
            detections.append(feature_vector)
        
        self.radar_data = np.array(detections)

    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()


class Obstacle_detector(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings):
        self.event = None
        
        obstacle_bp = blueprint_library.find('sensor.other.obstacle')

        obstacle_bp.set_attribute('distance', f'{sensor_settings[3]}')
        obstacle_bp.set_attribute('hit_radius', f'{sensor_settings[4]}')
        obstacle_bp.set_attribute('only_dynamics', 'true' if sensor_settings[5] == True else 'false')
        obstacle_bp.set_attribute('sensor_tick', f'{sensor_settings[6]}')
        transform = carla.Transform(carla.Location(x=sensor_settings[0], y=sensor_settings[1], z=sensor_settings[2]))

        self.sensor = world.spawn_actor(obstacle_bp, transform, attach_to=vehicle)

        self.sensor.listen(self.callback)
        
        
        

    def callback(self, event):
        self.event = event

    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()
                
class Semantic_segmentation_camera(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings, frametimes=None, testing=False):
        
        self.image = None
        self.img_width = sensor_settings[0]
        self.img_height = sensor_settings[1]
        self.frametimes = frametimes
        self.testing = testing

        semantic_seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_seg_bp.set_attribute('image_size_x', f'{self.img_width}')
        semantic_seg_bp.set_attribute('image_size_y', f'{self.img_height}')
        semantic_seg_bp.set_attribute('fov', f'{sensor_settings[5]}')
        semantic_seg_bp.set_attribute('sensor_tick', f'{sensor_settings[6]}')
        semantic_seg_bp.set_attribute('lens_circle_falloff', f'{sensor_settings[7]}')
        semantic_seg_bp.set_attribute('lens_circle_multiplier', f'{sensor_settings[8]}')
        semantic_seg_bp.set_attribute('lens_k', f'{sensor_settings[9]}')
        semantic_seg_bp.set_attribute('lens_kcube', f'{sensor_settings[10]}')
        semantic_seg_bp.set_attribute('lens_x_size', f'{sensor_settings[11]}')
        semantic_seg_bp.set_attribute('lens_y_size', f'{sensor_settings[12]}')
        transform = carla.Transform(carla.Location(x=sensor_settings[2], y=sensor_settings[3], z=sensor_settings[4]))
        
        self.sensor = world.spawn_actor(semantic_seg_bp, transform, attach_to=vehicle)
        
        self.sensor.listen(self.callback)

    def callback(self, image):
        image = np.array(image.raw_data)
        image = image.reshape((self.img_height, self.img_width, 4))
        image = image[:, :, :3]
        self.image = image

        if self.frametimes is not None:
            self.last_cam_update = time.time()
            if self.testing:
                self.frametimes.append(time.time() - self.last_cam_update)
            else:
                self.frametimes.put_nowait(time.time() - self.last_cam_update)


    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()
                
    
class Instance_segmentation_camera(SensorInterface):
    
    def __init__(self, world, blueprint_library, vehicle, sensor_settings, frametimes=None, testing=False):
        
        self.image = None
        self.img_width = sensor_settings[0]
        self.img_height = sensor_settings[1]
        self.frametimes = frametimes
        self.testing = testing
        
        instance_seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        instance_seg_bp.set_attribute('image_size_x', f'{self.img_width}')
        instance_seg_bp.set_attribute('image_size_y', f'{self.img_height}')
        instance_seg_bp.set_attribute('fov', f'{sensor_settings[5]}')
        instance_seg_bp.set_attribute('sensor_tick', f'{sensor_settings[6]}')
        instance_seg_bp.set_attribute('lens_circle_falloff', f'{sensor_settings[7]}')
        instance_seg_bp.set_attribute('lens_circle_multiplier', f'{sensor_settings[8]}')
        instance_seg_bp.set_attribute('lens_k', f'{sensor_settings[9]}')
        instance_seg_bp.set_attribute('lens_kcube', f'{sensor_settings[10]}')
        instance_seg_bp.set_attribute('lens_x_size', f'{sensor_settings[11]}')
        instance_seg_bp.set_attribute('lens_y_size', f'{sensor_settings[12]}')
        transform = carla.Transform(carla.Location(x=sensor_settings[2], y=sensor_settings[3], z=sensor_settings[4]))
        
        self.sensor = world.spawn_actor(instance_seg_bp, transform, attach_to=vehicle)
        
        self.sensor.listen(self.callback)
        
        
        

    def callback(self, image):
        image = np.array(image.raw_data)
        image = image.reshape((self.img_height, self.img_width, 4))
        image = image[:, :, :3]
        self.image = image

        if self.frametimes is not None:
            self.last_cam_update = time.time()
            if self.testing:
                self.frametimes.append(time.time() - self.last_cam_update)
            else:
                self.frametimes.put_nowait(time.time() - self.last_cam_update)

    def destroy(self):
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()


class GNSS_sensor(SensorInterface):
    """Responsible for everything regarding navigation as well. """
    def __init__(self, world, blueprint_library, vehicle, sensor_settings):
        self.data = None

        # trajectorie
        self.route = None
        self.spline_x = None
        self.spline_y = None
        self.distances = None

        # logging information
        self.waypoint_idx = 1
        self.skipped_waypoints = 0
        self.lateral_errors = []
        self.yaw_differences = []
        self.speeds = []

        gnss_bp = blueprint_library.find("sensor.other.gnss")
        gnss_bp.set_attribute('noise_alt_bias', f'{sensor_settings[0]}')
        gnss_bp.set_attribute('noise_alt_stddev', f'{sensor_settings[1]}')
        gnss_bp.set_attribute('noise_lat_bias', f'{sensor_settings[2]}')
        gnss_bp.set_attribute('noise_lat_stddev', f'{sensor_settings[3]}')
        gnss_bp.set_attribute('noise_lon_bias', f'{sensor_settings[4]}')
        gnss_bp.set_attribute('noise_lon_stddev', f'{sensor_settings[5]}')
        gnss_bp.set_attribute('noise_seed', f'{sensor_settings[6]}')
        gnss_bp.set_attribute('sensor_tick', f'{sensor_settings[7]}')
        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self.callback)


    def callback(self, data):
        self.data = data

    
    def generate_route(self, world, vehicle=None, debug=True):
        """Generates route based on waypoints."""
        if vehicle:
            vehicle_location = vehicle.get_location()
        else:
            vehicle_location = self.data.transform.location
        current_waypoint = world.get_map().get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        route = [current_waypoint]

        current_waypoint = current_waypoint.next(settings.DISTANCE_BETWEEN_WAYPOINTS)[0]
        
        accumulated_distance = 0.0

        # adds waypoints around every x meters until it reached a total distance
        while accumulated_distance < settings.DISTANCE_TO_GOAL:
            route.append(current_waypoint)
            next_waypoints = current_waypoint.next(settings.DISTANCE_BETWEEN_WAYPOINTS)

            if len(next_waypoints) == 0:
                break
        
            next_waypoint = next_waypoints[0]  
            distance_to_next = current_waypoint.transform.location.distance(next_waypoint.transform.location)

            current_waypoint = next_waypoint
            accumulated_distance += distance_to_next

        
        if debug:
            for waypoint in route:
                self.draw_waypoint(world, waypoint.transform.location)

        self.route = route
        self.create_polynomial()
        


    
    def draw_waypoint(self, world, waypoint_location, visualization_life_time=30):
        """Visualization of the Waypoints including their Radius."""

        # draw waypoint
        world.debug.draw_point(
                waypoint_location,
                size=0.1,  
                color=carla.Color(255, 0, 0),  
                life_time=visualization_life_time
            )
        
        # calculate and draw waypoint radius
        radius = settings.WAYPOINT_RADIUS
        num_segments = 36
        color=carla.Color(0, 255, 0)
        
        angle_increment = 2 * math.pi / num_segments
        points = []

        for i in range(num_segments):
            angle = i * angle_increment
            x = waypoint_location.x + radius * math.cos(angle)
            y = waypoint_location.y + radius * math.sin(angle)
            z = waypoint_location.z
            points.append(carla.Location(x=x, y=y, z=z))

        for i in range(num_segments):
            start_point = points[i]
            end_point = points[(i + 1) % num_segments]  
            world.debug.draw_line(
                start_point,
                end_point,
                thickness=0.1,
                color=color,
                life_time=visualization_life_time
            )


    
    def is_vehicle_in_waypoint_radius(self, vehicle=None):
        """Check if the vehicle is in the radius for the next waypoint. Also checks for goal reached and distributes navigation rewards."""
        reward = 0

        distance_to_next_waypoint, yaw_difference, orientation_difference = self.calculate_waypoint_alignment_errors(vehicle)

        distance_to_next_waypoint = abs(distance_to_next_waypoint)

        # means we reached a waypoint
        if distance_to_next_waypoint <= settings.WAYPOINT_RADIUS:
            self.waypoint_idx += 1
            reward = settings.WAYPOINT_REACHED_REWARD

            # lessens reward based on yaw difference to the waypoint
            if settings.REWARD_FUNCTION_METRICS['waypoint_yaw']:
                yaw_penalty = abs(yaw_difference) / math.pi
                reward *= (1 - yaw_penalty)
        

        else:
            # check if we skipped a waypoint
            if self.waypoint_idx < len(self.route) - 1:
                if vehicle:
                    vehicle_location = vehicle.get_transform().location
                else:
                    vehicle_location = self.data.transform.location
                distance_to_second_next_waypoint = vehicle_location.distance(self.route[self.waypoint_idx+1].transform.location)
                if distance_to_second_next_waypoint < distance_to_next_waypoint:
                    reward = -settings.WAYPOINT_MISSED_PENALTY
                    self.waypoint_idx += 1
                    self.skipped_waypoints += 1
                    distance_to_next_waypoint = distance_to_second_next_waypoint
                    
        
        # check if goal has been reached
        if self.waypoint_idx >= len(self.route) -1:
            reward += settings.GOAL_REACHED_REWARD
            return reward, True
        
        return reward, False
    

    def get_next_waypoint(self):
        if self.waypoint_idx == len(self.route):
            return None

        return self.route[self.waypoint_idx]
    


    def calculate_waypoint_alignment_errors(self, vehicle=None, angle_normalization=True):    
        """Calculates signed distance, yaw difference and optionally orientation_difference."""
        if self.waypoint_idx >= len(self.route):
            return 0, 0, 0

        # get vehicle's current position and yaw
        if vehicle:
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_yaw = vehicle_transform.rotation.yaw
        else:
            vehicle_location = self.data.transform.location
            vehicle_yaw = self.data.transform.rotation.yaw


        next_waypoint_location = self.route[self.waypoint_idx].transform.location
        next_waypoint_rotation = self.route[self.waypoint_idx].transform.rotation

        relative_position = carla.Location(
            x=next_waypoint_location.x - vehicle_location.x,
            y=next_waypoint_location.y - vehicle_location.y
        )

        distance_to_next_waypoint = relative_position.length()

        
        # compute the sign based on relative position
        vehicle_to_waypoint_vector = (next_waypoint_location.x - vehicle_location.x, 
                                    next_waypoint_location.y - vehicle_location.y)
        vehicle_heading_vector = (math.cos(math.radians(vehicle_yaw)), 
                                math.sin(math.radians(vehicle_yaw)))
        
        cross_product = (vehicle_heading_vector[0] * vehicle_to_waypoint_vector[1] -
                        vehicle_heading_vector[1] * vehicle_to_waypoint_vector[0])
        signed_distance_to_next_waypoint = math.copysign(distance_to_next_waypoint, cross_product)

        angle_to_waypoint = math.atan2(relative_position.y, relative_position.x)

        orientation_difference =  angle_to_waypoint - math.radians(vehicle_yaw)
        yaw_difference = math.radians(next_waypoint_rotation.yaw - vehicle_yaw)

        # normalize the yaw error to the range [-pi, pi]
        if angle_normalization:
            orientation_difference = (orientation_difference + math.pi) % (2 * math.pi) - math.pi
            yaw_difference = (yaw_difference + math.pi) % (2 * math.pi) - math.pi

        signed_distance_to_next_waypoint = round(signed_distance_to_next_waypoint, 2)
        orientation_difference = round(orientation_difference, 2)
        yaw_difference = round(yaw_difference, 2)

        return signed_distance_to_next_waypoint, yaw_difference, orientation_difference


    
    def create_polynomial(self):
        """Creates the polynomial based on the route waypoints."""
        route = self.route
        waypoints_x = [waypoint.transform.location.x for waypoint in route]
        waypoints_y = [waypoint.transform.location.y for waypoint in route]

        # sum up accurate length of route
        distances = np.zeros(len(waypoints_x))
        distances = np.insert(np.cumsum(np.sqrt(np.diff(waypoints_x)**2 + np.diff(waypoints_y)**2)), 0, 0)

        # create Cubic Splines based on Waypoints (trajectory)
        spline_x = CubicSpline(distances, waypoints_x)
        spline_y = CubicSpline(distances, waypoints_y)

        self.distances = distances
        self.spline_x = spline_x
        self.spline_y = spline_y


    
    def calculate_lane_alignment_errors(self, vehicle=None, angle_normalization=True):
        """Check vehicle position and rotation erros to the current closes point on the trajectorie. """
    
        idx = self.waypoint_idx
        
        if idx >= len(self.route):
            return 0,0,0
        
        prev_idx = max(0, idx - 1)
        next_idx = min(len(self.route) - 1, idx)

        # get vehicle's current position and yaw
        if vehicle:
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_yaw = vehicle_transform.rotation.yaw 
        else:
            vehicle_location = self.data.transform.location
            vehicle_yaw = self.data.transform.rotation.yaw  

        start_distance = self.distances[prev_idx]
        end_distance = self.distances[next_idx]

        # sample points on the spline only within this segment of the current and next waypoint
        sampled_distances = np.linspace(start_distance, end_distance, 50)  # 50 samples in segment
        x_samples = self.spline_x(sampled_distances)
        y_samples = self.spline_y(sampled_distances)

        # calculate distances from the vehicle's position to each sampled point
        distances_to_spline = np.sqrt((x_samples - vehicle_location.x)**2 + (y_samples - vehicle_location.y)**2)

        # find the closest point and its distance
        min_index = np.argmin(distances_to_spline)
        min_distance = distances_to_spline[min_index]
        closest_x = x_samples[min_index]
        closest_y = y_samples[min_index]
        
        # calculate the tangent angle (spline direction) at the closest point
        spline_tangent = float(np.arctan2(self.spline_y.derivative()(sampled_distances[min_index]),
                                    self.spline_x.derivative()(sampled_distances[min_index])))
        
        perpendicular_x = -math.sin(spline_tangent)
        perpendicular_y = math.cos(spline_tangent)

        vehicle_to_lane_center_x = closest_x - vehicle_location.x
        vehicle_to_lane_center_y = closest_y - vehicle_location.y

        # cross product to determine the sign
        cross_product = (vehicle_to_lane_center_x * perpendicular_y -
                        vehicle_to_lane_center_y * perpendicular_x)

        signed_min_distance = math.copysign(min_distance, cross_product)
        
        angle_to_lane_center = np.arctan2(closest_y - vehicle_location.y, closest_x - vehicle_location.x)
        orientation_difference = angle_to_lane_center - math.radians(vehicle_yaw)

        yaw_difference = spline_tangent - math.radians(vehicle_yaw)
        
        
        # normalize the yaw error to the range [-pi, pi]
        if angle_normalization:
            yaw_difference = (yaw_difference + np.pi) % (2 * np.pi) - np.pi
            orientation_difference = (orientation_difference + math.pi) % (2 * math.pi) - math.pi


        signed_min_distance = round(signed_min_distance, 2)
        yaw_difference = round(yaw_difference, 2)
        orientation_difference = round(orientation_difference, 2)

        return signed_min_distance, yaw_difference, orientation_difference
        

    
    def get_lane_center_rewards(self, vehicle):
        """Return rewards and check for end criteriums for lateral error and heading error."""

        distance, yaw_difference, _ = self.calculate_lane_alignment_errors(vehicle)
        distance = abs(distance)

        self.lateral_errors.append(distance)
        self.yaw_differences.append(abs(yaw_difference))

        
        reward = 0        

        # rewards / penaltys based on lateral distance to trajectorie
        if distance > settings.MAX_DISTANCE_BEFORE_ROUTE_LEFT or yaw_difference > settings.MAX_YAW_ERROR_BEFORE_ROUTE_LEFT:
            return -settings.ROUTE_LEFT_PENALTY, True

        elif distance < settings.MAX_DISTANCE_FOR_LANE_CENTER_REWARD:
            reward += settings.LANE_CENTER_MAX_REWARD * (1 - distance / settings.MAX_DISTANCE_FOR_LANE_CENTER_REWARD)
            # reward += settings.LANE_CENTER_MAX_REWARD

        else:
            distance_penalty = ((distance - settings.MAX_DISTANCE_FOR_LANE_CENTER_REWARD) / 
                                (settings.DISTANCE_FOR_MAX_LANE_CENTER_PENALTY - settings.MAX_DISTANCE_FOR_LANE_CENTER_REWARD))
            reward -= min(settings.LANE_CENTER_MAX_PENALTY, 
                          distance_penalty * settings.LANE_CENTER_MAX_PENALTY)

        # Reward/Penalty based on yaw error
        if abs(yaw_difference) < settings.MAX_YAW_ERROR_THRESHOLD:
            reward += settings.YAW_ERROR_MAX_REWARD * (1 - abs(yaw_difference) / settings.MAX_YAW_ERROR_THRESHOLD)
            #reward += settings.YAW_ERROR_MAX_REWARD
        else:
            # scale down to minimum reward based on yaw error
            yaw_penalty = ((abs(yaw_difference) - settings.MAX_YAW_ERROR_THRESHOLD) / 
                           (settings.YAW_PENALTY_ERROR_MAX - settings.MAX_YAW_ERROR_THRESHOLD))
            reward -= min(settings.YAW_ERROR_MAX_PENALTY, 
                          yaw_penalty * settings.YAW_ERROR_MAX_PENALTY)

        return reward, False
            

    def compute_curvature(self, s):
        """Computes curvuture, needed to calculate correct s for mpc state."""
        # first derivatives
        dx = self.spline_x(s, 1)
        dy = self.spline_y(s, 1)
        
        # second derivatives
        ddx = self.spline_x(s, 2)
        ddy = self.spline_y(s, 2)
        
        # curvature formula
        curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-8)  # avoid division by zero
        curvature = np.clip(curvature, -1e3, 1e3)  # clip extreme values

        return curvature    
    
    def get_close_vehicles():
        ...
    
    def get_traffic_light_state():
        ...
        
    def destroy(self):
        
        if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
            self.sensor.stop()
        if hasattr(self.sensor, 'is_alive') and self.sensor.is_alive:
                self.sensor.destroy()