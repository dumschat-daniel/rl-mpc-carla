from .helper import STOP, STOP_MESSAGE
from . import control_physics
from .sensors import *
from .mpc import MPC
from . import control_physics
from .mpc_to_carla import MPCToCarla
from .carla import CarlaEnv, CarlaEnvSettings, start as start_carla, restart as restart_carla, kill_carla_processes as kill_carla, ACTIONS, ACTIONS_NAMES, CARLA_SETTINGS_STATE, CARLA_SETTINGS_STATE_MESSAGE, set_sync_mode as set_carla_sync_mode, get_fixed_delta_seconds, Synchronizer
from .logging_tensorboard import TensorB
from .logging_neptuneAI import NeptuneLogger
from .agent import ARTDQNAgent, AGENT_STATE, AGENT_STATE_MESSAGE, run as run_agent, test as test_agent
from .experienceReplay import Reward_priorisation_experience_replay, PrioritizedReplayMemory
from .trainer import ARTDQNTrainer, TRAINER_STATE, TRAINER_STATE_MESSAGE, run as run_trainer, get_hparams, check_weights_size
from .console import ConsoleStats
from .commands import Commands
from .noise import GaussianNoise, OUNoise
from . import models


import settings

# quick setup
quick_setup = settings.QUICK_SETUP
 
if quick_setup is not None:
    settings.USE_N_FUTURE_STEPS = 1
    settings.PUT_TRANS_EVERY = 1
    settings.SYNC_MODE = True 
    settings.FIXED_DELTA_SEC = 0.1 
    settings.STEPS_PER_EPISODE = 500
    settings.AGENT_SYNCED_WITH_FRAME = False 
    settings.MINIBATCH_SIZE = 64 
    settings.TRAINING_BATCH_SIZE = 64
    settings.PREDICTION_BATCH_SIZE = 64
    settings.UPDATE_TARGET_EVERY = 50
    settings.USE_N_FUTURE_STEPS = 1
    settings.PUT_TRANS_EVERY = 1 
    settings.DISCOUNT = 0.99
    settings.EXPERIENCE_REPLAY_METHOD = 'reward_old'
    settings.EXPERIENCE_REPLAY_SIZE = 50_000
    settings.MIN_EXPERIENCE_REPLAY_SIZE = 500 
    settings.MODEL_INPUTS = {'front_camera': False, 'lidar': False, 'radar': False, 'relative_pos': True, 'relative_orientation': False, 'collision': False, 'lane_invasion': False, 'speed': True, 'last_action': False, 'last_agent_input': False, 'distance_to_lane_center': True, 'orientation_difference_to_lane_center': False, 'close_vehicles': False, 'acceleration': False, 'yaw_angle': False, 'jerk_rate': False, 'traffic_light_state': False, 'obstacle': False}
    settings.TAU = 0.001 
    settings.ACTOR_GRADIENT_CLIP_NORM = 1 
    settings.CRITIC_GRADIENT_CLIP_NORM = 5 
    settings.USE_MPC = False
    settings.RL_ALGORITHM = quick_setup
    settings.DDPG_ACTOR_OPTIMIZER_LEARNING_RATE = 1e-5
    settings.DDPG_CRITIC_OPTIMIZER_LEARNING_RATE = 1e-4


if quick_setup == 'dqn':
    settings.ACTIONS = ['forward_slow', 'forward_medium', 'left_slow', 'left_medium', 'right_slow', 'right_medium']
    settings.START_EPSILON = 1.0
    settings.EPSILON_DECAY = 0.99975 
    settings.MIN_EPSILON = 0
    settings.DQN_OPTIMIZER_LEARNING_RATE = 0.001
    settings.DQN_OPTIMIZER_DECAY = 0.0

else:
    settings.DDPG_PREPROCESS_ACTION_INPUT = True 
    settings.SCALE_THROTTLE = [-1, 1] 
    settings.SCALE_STEER = [-1, 1] 
    settings.NOISE_DECAY_STRATEGY = 'exponential' 
    settings.NOISE_TYPE = 'OUNoise'
    settings.NOISE_THROTTLE_PARAMS = {'mu': 0.1, 'theta': 0.2, 'sigma': 0.4, 'min_sigma': 0.05, 'sigma_decay': 0.9999}
    settings.NOISE_STEERING_PARAMS = {'mu': 0, 'theta': 0.2, 'sigma': 0.25, 'min_sigma': 0.05, 'sigma_decay': 0.9999} 
    settings.DDPG_ACTOR_OPTIMIZER_LEARNING_RATE = 1e-4
    settings.DDPG_ACTOR_OPTIMIZER_DECAY = 0.999
    settings.DDPG_CRITIC_OPTIMIZER_LEARNING_RATE = 1e-3
    settings.DDPG_CRITIC_OPTIMIZER_DECAY = 0.999
    settings.DDPG_LR_DECAY = False
    settings.TD3_SMOOTHING_STD = 0.2 
    settings.TD3_NOISE_CLIP = 0.3
    settings.TD3_DELAYED_POLICY_UPDATE = 10 
    

if quick_setup in ['phase-1','phase-2','phase-3']:
    settings.RL_ALGORITHM = 'ddpg'
    settings.MIN_EXPERIENCE_REPLAY_SIZE = 5_000
    settings.NOISE_TYPE = 'GaussianNoise'
    settings.NOISE_THROTTLE_PARAMS = {'mu': 0.0, 'theta': 0.2, 'sigma': 0.2, 'min_sigma': 0.05, 'sigma_decay': 0.999} 
    settings.NOISE_STEERING_PARAMS = {'mu': 0.0, 'theta': 0.2, 'sigma': 0.2, 'min_sigma': 0.05, 'sigma_decay': 0.999} 

    settings.DDPG_ACTOR_OPTIMIZER_LEARNING_RATE = 1e-4
    settings.DDPG_CRITIC_OPTIMIZER_LEARNING_RATE = 1e-3
    if quick_setup == 'phase-1':
        settings.USE_MPC = True
        settings.MPC_PHASE = 'Imitation'

    elif quick_setup == 'phase-2':
        settings.USE_MPC = True
        settings.MPC_CRITIC_START_EPSILON = 1.0 
        settings.MPC_CRITIC_EPSILON_DECAY = 0.9995 
        settings.MPC_CRITIC_EPSILON_MIN = 0.0 
        settings.MPC_EXPLORATION_START_EPSILON = 0.25
        settings.MPC_EXPLORATION_EPSILON_DECAY = 0.95 
        settings.MPC_EXPLORATION_MIN_EPSILON = 0.25 


# Setting up sensor variables
if not settings.SENSOR_DEFAULT_SETTINGS:
    settings.RGB_CAM_SETTINGS = [settings.RGB_CAM_IMG_WIDTH, settings.DEPTH_CAM_IMG_HEIGHT, *settings.RGB_CAM_POS, settings.RGB_CAM_FOV, settings.RGB_CAM_IMG_TYPE, settings.RGB_BLOOM_INTENSITY, settings.RGB_FSTOP, settings.RGB_ISO, settings.RGB_GAMMA, settings.RGB_LENS_FLARE_INTENSITY, settings.RGB_SENSOR_TICK, settings.RGB_SHUTTER_SPEED, settings.RGB_LENS_CIRCLE_FALLOFF, settings.RGB_LENS_CIRCLE_MULTIPLIER, settings.RGB_LENS_K, settings.RGB_LENS_KCUBE, settings.RGB_LENS_X_SIZE, settings.RGB_LENS_Y_SIZE]
    settings.DEPTH_CAM_SETTINGS = [settings.DEPTH_CAM_IMG_WIDTH, settings.DEPTH_CAM_IMG_HEIGHT, *settings.DEPTH_CAM_POS, settings.DEPTH_CAM_FOV, settings.DEPTH_CAM_TICK, settings.DEPTH_LENS_CIRCLE_FALLOFF, settings.DEPTH_LENS_CIRCLE_MULTIPLIER, settings.DEPTH_LENS_K, settings.DEPTH_LENS_KCUBE, settings.DEPTH_LENS_X_SIZE, settings.DEPTH_LENS_Y_SIZE ]
    settings.SEMANTIC_CAM_SETTINGS = [settings.SEMANTIC_CAM_IMG_WIDTH, settings.SEMANTIC_CAM_IMG_HEIGHT, *settings.SEMANTIC_CAM_POS, settings.SEMANTIC_CAM_FOV, settings.SEMANTIC_SENSOR_TICK, settings.SEMANTIC_LENS_CIRCLE_FALLOFF, settings.SEMANTIC_LENS_CIRCLE_MULTIPLIER, settings.SEMANTIC_LENS_K, settings.SEMANTIC_LENS_KCUBE, settings.SEMANTIC_LENS_X_SIZE, settings.SEMANTIC_LENS_Y_SIZE]
    settings.INSTANCE_CAM_SETTINGS = [settings.INSTANCE_CAM_IMG_WIDTH, settings.INSTANCE_CAM_IMG_HEIGHT, *settings.INSTANCE_CAM_POS, settings.INSTANCE_CAM_FOV, settings.INSTANCE_SENSOR_TICK, settings.INSTANCE_LENS_CIRCLE_FALLOFF, settings.INSTANCE_LENS_CIRCLE_MULTIPLIER, settings.INSTANCE_LENS_K, settings.INSTANCE_LENS_KCUBE, settings.INSTANCE_LENS_X_SIZE, settings.INSTANCE_LENS_Y_SIZE]
    settings.IMU_SETTINGS = [*settings.IMU_POS, settings.IMU_SENSOR_TICK, settings.IMU_NOISE_ACCEL_STDDEV_X, settings.IMU_NOISE_ACCEL_STDDEV_Y, settings.IMU_NOISE_ACCEL_STDDEV_Z, settings.IMU_NOISE_GYRO_BIAS_X, settings.IMU_NOISE_GYRO_BIAS_Y, settings.IMU_NOISE_GYRO_BIAS_Z, settings.IMU_NOISE_GYRO_STDDEV_X, settings.IMU_NOISE_GYRO_STDDEV_Y, settings.IMU_NOISE_GYRO_STDDEV_Z, settings.IMU_NOISE_SEED]
    settings.LANE_INVASION_SETTINGS = [*settings.LANE_INVASION_DETECTOR_POS]
    settings.RADAR_SETTINGS = [*settings.RADAR_POS, settings.RADAR_FOV_HOR, settings.RADAR_FOV_VER, settings.RADAR_RANGE, settings.RADAR_PPS, settings.RADAR_TICK]
    settings.LIDAR_SETTINGS = [*settings.LIDAR_POS, settings.LIDAR_PREPROCESSING_METHOD, settings.LIDAR_CHANNELS, settings.LIDAR_RANGE, settings.LIDAR_POINTS_PER_SECOND, settings.LIDAR_ROTATION_FREQUENCY, settings.LIDAR_UPPER_FOV, settings.LIDAR_LOWER_FOV, settings.LIDAR_HORIZONTAL_FOV, settings.LIDAR_ATMOSPHERE_ATTENUATION_RATE, settings.LIDAR_DROPOFF_GENERAL_RATE, settings.LIDAR_DROPOFF_INTENSITY_LIMIT, settings.LIDAR_DROPOFF_ZERO_INTENSITY, settings.LIDAR_SENSOR_TICK]
    settings.OBSTACLE_DETECTOR_SETTINGS = [*settings.OBSTACLE_DETECTOR_POS, settings.OBSTACLE_DETECTOR_DISTANCE, settings.OBSTACLE_DETECTOR_HIT_RADIUS, settings.OBSTACLE_DETECTOR_ONLY_DYNAMICS, settings.OBSTACLE_DETECTOR_TICK]
    settings.GNSS_SENSOR_SETTINGS = [settings.GNSS_NOISE_ALT_BIAS, settings.GNSS_NOISE_ALT_STDDEV, settings.GNSS_NOISE_LAT_BIAS, settings.GNSS_NOISE_LAT_STDDEV, settings.GNSS_NOISE_LON_BIAS, settings.GNSS_NOISE_LON_STDDEV, settings.GNSS_NOISE_SEED, settings.GNSS_SENSOR_TICK]


main_cam = settings.FRONT_CAM_TYPE
 
if main_cam == 'depth':

    settings.FRONT_CAM_SETTINGS = settings.DEPTH_CAM_SETTINGS

elif main_cam == 'semseg':
    
    settings.FRONT_CAM_SETTINGS = settings.SEMANTIC_CAM_SETTINGS

elif main_cam == 'inseg':

    settings.FRONT_CAM_SETTINGS = settings.INSTANCE_CAM_SETTINGS

else:

    settings.FRONT_CAM_SETTINGS = settings.RGB_CAM_SETTINGS


if models.MODEL_NAME_PREFIX:
    settings.MODEL_NAME = models.MODEL_NAME_PREFIX + ('_' if models.MODEL_NAME else '') + settings.MODEL_NAME





