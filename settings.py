QUICK_SETUP = 'ddpg' # phase-1, phase-2, phase-3, dqn, ddpg, td3 # Sets up most important parameters to default settings. (Phase 1,2,3 uses ddpg)

# Carla setup
CARLA_PATH = 'C:/Users/ddums/Desktop/WindowsNoEditor' # path to your carla installation 
CARLA_HOSTS_TYPE = 'local'
DONT_RENDER_VISUALS = True # last parameter of carla hosts, if yes it doesn't render visuals which speeds up training. Visuals Sensors won't work.
CARLA_HOSTS = [['localhost', 2000, 'Town10HD_Opt', False, False, DONT_RENDER_VISUALS]] # host, port (first of 2 consecutive ports), map, fps benchmark, quality level, no render mode
CARLA_HOSTS_NO = 1 # amount of carla hosts we have. Default 1
FREE_OCCUPIED_PORT = True # if something is running on the desired Carla Port then kill that process
SETUP_TIME = 0.7 # 0.7 # this might depend on your hardware. When spawning an actor it takes some time until the actor accepts input (for me its about 0.5secs and I gave it some leeway). Refer to the helper setup_time.py.


# Sync Mode Settings
'''
In general Sync Mode should be enabled for Training and Validation. MPC only works in sync mode.
'''
SYNC_MODE = True # Sync Mode is prefered for Training. Client controls the simulation. False = Async Mode
FIXED_DELTA_SEC = 0.1 # by how much to advance the Simulation on every tick in sync mode
STEPS_PER_EPISODE = 500 # max Steps per Episode (end criterium for sync mode)
SECONDS_PER_EPISODE = 30 # max len of every episode (end criterium for async mode)

# Carla env settings
MAP_LAYER_GROUND_ONLY = True # This disables all Map layers but the ground, keeps the Training Environment uniform
CAR_NPCS = 0 # amount of other vehicles in the Simulation
SPEED_FACTOR = 1.0 # Speed factor of the environment Settings (Weather etc.) for async mode
RESET_CAR_NPC_EVERY_N_TICKS = 300  # Resets one car NPC every n ticks. Safety mechanism if NPC Vehicles get stuck and block traffic
ROTATE_MAP_EVERY = False # in minutes or False for no map rotation 
WEATHER_PRESET = "ClearSunset" 
DYNAMIC_WEATHER = False # Changes the Weather dynamicly with every tick
DISALLOWED_NPC_VEHICLES = ['vehicle.mitsubishi.fusorosa', 'vehicle.carlamotors.carlacola'] # disallow some trouble makers


# Agent Settings
AGENTS = 2 # multi agent training
AGENT_SYNCED_WITH_FRAME = False  # Synchronizes agent with frame updates from Carla in async mode to make sure that agents don't predict on the same frame twice when agent fps > carla fps
VEHICLE = 'vehicle.tesla.model3' # hero vehicle. For other vehicles settings like Wheelbase or Sensors might have to be adjusted
AGENT_CARLA_INSTANCE = [] # only relevant if you use more than 1 carla server. Can be used to put agents on a certain carla server instance.
AGENT_SHOW_PREVIEW = [] # # List of agent id's to show a preview, or empty list
EPISODE_FPS = 30  # Desired. used for weighted rewards in async mode


# rewards / Penaltys
REWARD_RANGE = None # Limit the reward range per step.

# Reward function metrics contains all possible factors for the reward function. True if use
REWARD_FUNCTION_METRICS = {'move': False, 'collision': True, 'speed': True, 'jerk': False, 'waypoint_reached': True, 'waypoint_yaw': False, 'lane_invasion': False, 'progress': False, 'lane_center': True, 'imu': False}
# Rewards / Penaltys. Penaltys are set as a positive value.
MOVE_PENALTY = 1 # Applied on every move
COLLISION_PENALTY = 2  # Applied on Collision 
WEIGHT_REWARDS_WITH_SPEED = 'area'  # 'discrete''linear': -1..1, 'quadratic', 'area' = speed is between min and max speed.
MIN_SPEED = 20 # Min Speed for reward
MAX_SPEED = 30 # Max Speed for reward
SPEED_MAX_REWARD = 0.5 # Reward if inside the speed area
SPEED_MAX_PENALTY = 1.0 # the maximum penalty for deviation to the target speed (mid of speed area)

# Navigation
DISTANCE_TO_GOAL = 200 # The total distance to goal for an episode
DISTANCE_BETWEEN_WAYPOINTS = 2 # the desired distance between waypoints
WAYPOINT_RADIUS = 0.3 # radius of the waypoints (inside the radius it counts as waypoint reached)
WAYPOINT_REACHED_REWARD = 0.5 # reward for reaching a waypoint
GOAL_REACHED_REWARD = 2 # reward for completing the episode (reaching last waypoint)
WAYPOINT_MISSED_PENALTY = 0.5 # penalty for missing a waypoint (if distance to 2nd next waypoint is smaller than to the next the waypoint will be skipped and the penalty will be applied)

MAX_DISTANCE_FOR_LANE_CENTER_REWARD = 0.3 # The max distance to lane center to get a reward
DISTANCE_FOR_MAX_LANE_CENTER_PENALTY = 3 # the distance for max lane center penalty
LANE_CENTER_MAX_REWARD = 2.0 # max reward if within the max distnace for lane center reward
LANE_CENTER_MAX_PENALTY = 2.0 # max penalty possible for lane center


MAX_DISTANCE_BEFORE_ROUTE_LEFT = 3 # end criterium. Max Deviation from the lane center 
MAX_YAW_ERROR_BEFORE_ROUTE_LEFT = 0.785 # ~45 degrees
ROUTE_LEFT_PENALTY = 2 # 2 # penalty for leaving the route

YAW_ERROR_MAX_REWARD = 1.0 # max reward for the allignment with the route curviture
YAW_ERROR_MAX_PENALTY = 1.0 # max penalty for the allignment with the route curviture
MAX_YAW_ERROR_THRESHOLD = 0.175  # radians (~10 degrees) # yaw error threshold for reward
YAW_PENALTY_ERROR_MAX = 0.78 # radians (~45 degrees) # yaw error threshold for max penalty

ADDITIONAL_PENALTYS = [] # speed_slow, speed_fast, steering # gives additional penaltys for those actions if learning is stuck in a local minimum. (Steering is abs > 0.5)

# Trainer settings
USE_HPARAMS = True # continue training from checkpoint
MINIBATCH_SIZE = 64 
TRAINING_BATCH_SIZE = 64
PREDICTION_BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 10 # Update target network every x iterations.
MIN_REWARD = 400 # initial min reward for model save
MIN_AVG_REWARD = 400 # initial min avg reward for model save
AGGREGATE_STATS_EVERY = 10 # for logging (stats of every x amount of episodes)
SAVE_CHECKPOINT_EVERY = 250 # every n episodes 
ADDITIONAL_SAVE_FOLDER_PATH = ''

RL_ALGORITHM = 'ddpg' # dqn, ddpg, td3
DISCOUNT = 0.99
UPDATE_WEIGHTS_EVERY = 0
USE_N_FUTURE_STEPS = 1 # multi step environment, how many future states should be used for every transition
PUT_TRANS_EVERY = 1 # how often to put an transition, should be <= USE_N_FUTURE_STEPS

# DQN settings
# the actions that the model can take. These get mapped in the carla class to actual carla actions
ACTIONS = ['forward_slow', 'forward_medium', 'left_slow', 'left_medium', 'right_slow', 'right_medium'] # ['forward_slow', 'forward_medium', 'forward_fast', 'left_slow', 'left_medium', 'left_fast', 'right_slow', 'right_medium', 'right_fast', 'brake_light', 'brake_medium', 'brake_full']
SMOOTH_ACTIONS = False # smooths out actions 
START_EPSILON = 1.0
EPSILON_DECAY = 0.99975 ## 0.9975 99975
MIN_EPSILON = 0
# Optimizer settings
DQN_OPTIMIZER_LEARNING_RATE = 0.001
DQN_OPTIMIZER_DECAY = 0.0

# DDPG / Td3 settings
DDPG_PREPROCESS_ACTION_INPUT = True # for critic network
SCALE_THROTTLE = [-1, 1] # Scale throttle to a certain range
SCALE_STEER = [-1, 1] # Scale steering to a certain range
# Noise
NOISE_DECAY_STRATEGY = 'exponential' # linear, exponential, logarithmic
NOISE_TYPE = 'GaussianNoise' # GaussianNoise, OUNoise
if NOISE_TYPE == 'OUNoise':
    NOISE_THROTTLE_PARAMS = {'mu': 0.0, 'theta': 0.2, 'sigma': 0.2, 'min_sigma': 0.05, 'sigma_decay': 0.9995} # 0.2, 0.05 0.99999
    NOISE_STEERING_PARAMS = {'mu': 0.0, 'theta': 0.2, 'sigma': 0.2, 'min_sigma': 0.05, 'sigma_decay': 0.9995} #0.99999
else:
    NOISE_THROTTLE_PARAMS = {'mu': 0.1, 'theta': 0.2, 'sigma': 0.4, 'min_sigma': 0.05, 'sigma_decay': 0.999}
    NOISE_STEERING_PARAMS = {'mu': 0, 'theta': 0.2, 'sigma': 0.25, 'min_sigma': 0.05, 'sigma_decay': 0.999} 


# Optimizer settings
DDPG_ACTOR_OPTIMIZER_LEARNING_RATE = 1e-4
DDPG_ACTOR_OPTIMIZER_DECAY = 0.999
DDPG_CRITIC_OPTIMIZER_LEARNING_RATE = 1e-3
DDPG_CRITIC_OPTIMIZER_DECAY = 0.999
DDPG_LR_DECAY = False

TD3_SMOOTHING_STD = 0.2 # for training noise
TD3_NOISE_CLIP = 0.3 # for training noise
TD3_DELAYED_POLICY_UPDATE = 2 # target network updates


# Experience Replay settings
EXPERIENCE_REPLAY_METHOD = 'reward_old' # td_error , td_error_reward, reward, td_error_reducible_loss, reward_old (older implementation of reward based experience replay)
EXPERIENCE_REPLAY_SIZE = 50_000  # How many last steps to keep for model training
MIN_EXPERIENCE_REPLAY_SIZE = 5_000 #5_000 # Minimum number of transitions in memory to start training
EXPERIENCE_REPLAY_ALPHA = 0.6
EXPERIENCE_REPLAY_BETA = 0.4
EXPERIENCE_REPLAY_BETA_SCALE = 1e-5 # how fast beta increases per batch
EXPERIENCE_REPLAY_LAMBDA = 0.75

# Model settings
MODEL_INPUTS = {'front_camera': False, 'lidar': False, 'radar': False, 'relative_pos': True, 'relative_orientation': False, 'collision': False, 'lane_invasion': False, 'speed': True, 'last_action': False, 'last_agent_input': False, 'distance_to_lane_center': True, 'orientation_difference_to_lane_center': False, 'close_vehicles': False, 'acceleration': False, 'yaw_angle': False, 'jerk_rate': False, 'traffic_light_state': False, 'obstacle': False}
MODEL_BASES = {} #{"camera": "resnet50", "lidar": "resnet50"} # model inputs that require more preprocessing
MODEL_NAME = '' # redundant currently
MODEL_INITIAL_BIAS = [0.5, 0] # set Initial Bias for last layer 
MODEL_NORMALIZATION = None # layer, batch, None
MODEL_CAMERA_BASE_SETTINGS = {'num_of_dense_layers': 2, 'num_of_outputs': 128}
MODEL_LIDAR_BASE_SETTINGS = {'num_of_dense_layers': 3, 'num_of_outputs': 128}
TAU = 0.001 # None
ACTOR_GRADIENT_CLIP_NORM = 1 #1 
CRITIC_GRADIENT_CLIP_NORM = 5 #5
# Conv Cam
CONV_CAM_LAYER = None  # 'auto' - finds and uses last activation layer, 'auto_act' - uses Activation layer after last convolution layer if exists
CONV_CAM_AGENTS = [] # id of the agent to use conv (starting at 1)


# MPC Settings
USE_MPC = False # If to enable MPC
MPC_PHASE = '' # Imitation, Transition. Only relevant if MPC is enabled, controls the exploration and loss function parameters
# Settings for Transition MPC Phase
MPC_CRITIC_START_EPSILON = 1.0 # Start Epsilon for Critic. Epsilon * MPC Loss + (1-Epsilon) * Critic Loss
MPC_CRITIC_EPSILON_DECAY = 0.9995 # Exponential Decay
MPC_CRITIC_EPSILON_MIN = 0.0 # MIN Epsilon for Critic
MPC_EXPLORATION_START_EPSILON = 0.25 # Start Epsilon for Exploration. Average chance for choosing MPC Action instead of Actor.
MPC_EXPLORATION_EPSILON_DECAY = 0.95 # Exponential Decay
MPC_EXPLORATION_MIN_EPSILON = 0.25 # Min Epsilon for Exploration


# MPC as safety measure on inference. Thresholds for MPC interference
MPC_LANE_CENTER_DISTANCE_THRESHOLD = MAX_DISTANCE_BEFORE_ROUTE_LEFT * 0.75
MPC_VELOCITY_THRESHOLD = MAX_SPEED * 1.25
MPC_YAW_DIFFERENCE_THRESHOLD = MAX_YAW_ERROR_BEFORE_ROUTE_LEFT * 0.75


'''
Sensor Settings
When using more than 1 Sensor of the same Type (Count > 1) then Values corresponding to that Sensor Type (when they differ) should be set as a List of the same length of count. If they are set as an Value,
then the Value will be used for all Instances. 
'''       

SENSOR_DEFAULT_SETTINGS = False # Sets the Sensor Settings to default values provided by Carla instead of the specified values (not implemented yet)
# Camera
FRONT_CAM_TYPE = 'rgb' # rgb, depth, semseg, inseg, 
IMG_WIDTH = 480
IMG_HEIGHT = 270
FOV = 110
CAM_POS = [2.5,0,1] # X, Y, Z camera pos relative to car
FRONT_CAM_SETTINGS = []
# RGB
RGB_CAM_IMG_TYPE = 'rgb' # rgb, grayscaled or stacked (stacks last 3 consecutive grayscaled frames)
RGB_CAM_IMG_WIDTH = IMG_WIDTH 
RGB_CAM_IMG_HEIGHT = IMG_HEIGHT
RGB_CAM_FOV = FOV
RGB_CAM_POS = CAM_POS
RGB_BLOOM_INTENSITY = 0.675
RGB_FSTOP = 1.4
RGB_ISO = 100.0
RGB_GAMMA = 2.2
RGB_LENS_FLARE_INTENSITY = 0.1
RGB_SENSOR_TICK = 0.0
RGB_SHUTTER_SPEED = 200.0
RGB_LENS_CIRCLE_FALLOFF = 5.0
RGB_LENS_CIRCLE_MULTIPLIER = 0.0
RGB_LENS_K = -1
RGB_LENS_KCUBE = 0.0
RGB_LENS_X_SIZE = 0.08
RGB_LENS_Y_SIZE = 0.08
RGB_CAM_SETTINGS = []
# Depth
DEPTH_CAM_IMG_WIDTH = IMG_WIDTH
DEPTH_CAM_IMG_HEIGHT = IMG_HEIGHT
DEPTH_CAM_FOV = FOV
DEPTH_CAM_POS = CAM_POS
DEPTH_CAM_TICK = 0.0
DEPTH_LENS_CIRCLE_FALLOFF = 5.0
DEPTH_LENS_CIRCLE_MULTIPLIER = 0.0
DEPTH_LENS_K = -1.0
DEPTH_LENS_KCUBE = 0.0
DEPTH_LENS_X_SIZE = 0.08
DEPTH_LENS_Y_SIZE = 0.08
DEPTH_CAM_SETTINGS = []

# Semantic
SEMANTIC_CAM_IMG_WIDTH = IMG_WIDTH
SEMANTIC_CAM_IMG_HEIGHT = IMG_HEIGHT
SEMANTIC_CAM_FOV = FOV
SEMANTIC_CAM_POS = CAM_POS
SEMANTIC_SENSOR_TICK = 0.0
SEMANTIC_LENS_CIRCLE_FALLOFF = 5.0
SEMANTIC_LENS_CIRCLE_MULTIPLIER = 0.0
SEMANTIC_LENS_K = -1
SEMANTIC_LENS_KCUBE = 0.0
SEMANTIC_LENS_X_SIZE = 0.08
SEMANTIC_LENS_Y_SIZE = 0.08
SEMANTIC_CAM_SETTINGS = []

# Instance
INSTANCE_CAM_IMG_WIDTH = IMG_WIDTH
INSTANCE_CAM_IMG_HEIGHT = IMG_HEIGHT
INSTANCE_CAM_FOV = FOV
INSTANCE_CAM_POS = CAM_POS
INSTANCE_SENSOR_TICK = 0.0
INSTANCE_LENS_CIRCLE_FALLOFF = 5.0
INSTANCE_LENS_CIRCLE_MULTIPLIER = 0.0
INSTANCE_LENS_K = -1
INSTANCE_LENS_KCUBE = 0.0
INSTANCE_LENS_X_SIZE = 0.08
INSTANCE_LENS_Y_SIZE = 0.08
INSTANCE_CAM_SETTINGS = []

# Preview 
PREVIEW_CAMERA_PRESETS = [[640, 400, 2.5,0,1, 110], [640, 400, -5, 0, 2.5, 110]] # width, height, x, y, z, fov
# Collision Sensor
COLLISION_FILTER = [['static.sidewalk', -1], ['static.road', -1], ['vehicle.', 500]] # if collision is significant enough to cause episode end (-1 means to not end the episode on that collision)
# IMU 
IMU_SENSOR_TICK = 0.0 # default 0.0 (= output as often as possible so every step). 
IMU_POS = [0,0,1] # X, Y, Z IMU pos relative to car
IMU_NOISE_SEED = 0
IMU_NOISE_ACCEL_STDDEV_X = 0.0
IMU_NOISE_ACCEL_STDDEV_Y = 0.0
IMU_NOISE_ACCEL_STDDEV_Z = 0.0
IMU_NOISE_GYRO_BIAS_X = 0.0
IMU_NOISE_GYRO_BIAS_Y = 0.0
IMU_NOISE_GYRO_BIAS_Z = 0.0
IMU_NOISE_GYRO_STDDEV_X = 0.0
IMU_NOISE_GYRO_STDDEV_Y = 0.0
IMU_NOISE_GYRO_STDDEV_Z = 0.0
IMU_SETTINGS = []

# Lane Invasion Detector 
LANE_INVASION_DETECTOR_POS = [0.5,0,1] # X, Y, Z lane invasion detector pos relative to car
LANE_INVASION_FILTER = [['SOLID', -5],['BROKEN', -1],['SOLID_SOLID', -5],['SOLID_BROKEN', -1],['BROKEN_SOLID', -1],['BROKEN_BROKEN', -1],['BOTTS_DOTS', -5],['CURB', -5],['GRASS', -5]]
LANE_INVASION_SETTINGS = []
# RADAR 
RADAR_POS = [2.0, 0, 0.5]
RADAR_FOV_HOR = 30
RADAR_FOV_VER = 30
RADAR_RANGE = 100
RADAR_PPS = 1500 
RADAR_TICK = 0.0 # default 0.0 (= output as often as possible so every step). 
RADAR_SETTINGS = []

# Lidar 
LIDAR_POS = [0,0,2]
LIDAR_CHANNELS = 32
LIDAR_RANGE = 10
LIDAR_POINTS_PER_SECOND = 56000
LIDAR_ROTATION_FREQUENCY = 10
LIDAR_UPPER_FOV = 10
LIDAR_LOWER_FOV = -30
LIDAR_HORIZONTAL_FOV = 360
LIDAR_ATMOSPHERE_ATTENUATION_RATE = 0.004
LIDAR_DROPOFF_GENERAL_RATE = 0.45
LIDAR_DROPOFF_INTENSITY_LIMIT = 0.8
LIDAR_DROPOFF_ZERO_INTENSITY = 0.4
LIDAR_SENSOR_TICK = 0.0
LIDAR_PREPROCESSING_METHOD = 'birds_eye_view' # point_cloud, voxel_grid, birds_eye_view, range_images
# birds eye view settings
BEV_GRID_SIZE = (512,512) # dimensions of BEV image
BEV_GRID_RESOLUTION = LIDAR_RANGE * 2 / BEV_GRID_SIZE[0] # how much real-world distance (in meters) each grid cell (pixel) covers (scaling factor)
BEV_Z_THRESHOLD = [-1.5, 2.0] # Lower end of what we want to cover in our bev image
BEV_CLAMP_POINTS = False
LIDAR_SETTINGS = []

# Obstacle Detector
OBSTACLE_DETECTOR_POS = [1,0,2]
OBSTACLE_DETECTOR_DISTANCE = 15
OBSTACLE_DETECTOR_HIT_RADIUS = 0.5
OBSTACLE_DETECTOR_ONLY_DYNAMICS = False
OBSTACLE_DETECTOR_TICK = 0.0 # default 0.0 (= output as often as possible so every step). 
OBSTACLE_DETECTOR_SETTINGS = []

# GNSS Sensor
GNSS_NOISE_ALT_BIAS = 0.0
GNSS_NOISE_ALT_STDDEV = 0.0
GNSS_NOISE_LAT_BIAS = 0.0
GNSS_NOISE_LAT_STDDEV = 0.0
GNSS_NOISE_LON_BIAS = 0.0
GNSS_NOISE_LON_STDDEV = 0.0
GNSS_NOISE_SEED = 0.0
GNSS_SENSOR_TICK = 0.0
GNSS_SENSOR_SETTINGS = []

# GPU
USE_GPU = True # use GPU for training
AGENT_GPU = None # set dedicated gpu for our agent. None means TF decides
TRAINER_GPU = None # set dedicated gpu for our trainer. None means TF decides
TRAINER_MEMORY = 12 * 1024 # set maximum memory for trainer
AGENT_MEMORY = 1 * 1024 / AGENTS # set maximum memory for each agent

# console
PRINT_CONSOLE = False # prints current training information to the command line
SHOW_CARLA_ENV_SETTINGS = False # include environment settings 
PRINT_CONSOLE_EVERY = 1.0 # prints every 

# Additional Logging
ADDITIONAL_LOGGING = [] # neptune
NEPTUNE_PROJECT_NAME = ""
NEPTUNE_API_TOKEN = ""


# Testing 
# Testing disables all end criteriums but the max steps, collision or the correct absolvation of the scenario
TESTING_CARLA_HOST = [['localhost', 2000, 'Town04_Opt', False, False, False]] # host, port (first of 2 consecutive ports), map, fps benchmark, quality level, no render mode
TESTING_ROUTE_AREA = [0,3] # using route 0 to 40 for testing
TESTING_MAX_STEPS = 500 # max steps before scenario counts as failed
TESTING_RL_ALGORITHM = "ddpg" # used rl algorithm
TESTING_PREVIEW = True # preview uses testing camera settings and only works if no render mode is False (last setting in host)
TESTING_CAMERA = [640, 400, 2.5, 0, 1, 110] # width, height, x, y, z, fov
