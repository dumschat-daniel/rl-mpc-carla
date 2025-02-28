from dataclasses import dataclass
import settings

# global states for communication between Processes 
@dataclass(frozen=True)
class STOP:
    running = 0
    now = 1
    at_checkpoint = 2
    stopping = 3
    stopped = 4
    carla_simulator_error = 5
    restarting_carla_simulator = 6
    carla_simulator_restarted = 7



STOP_MESSAGE = {
    0: 'RUNNING',
    1: 'STOP NOW',
    2: 'STOP AT CHECKPOINT',
    3: 'STOPPING',
    4: 'STOPPED',
    5: 'CARLA SIMULATOR ERROR',
    6: 'RESTART' + ('ING' if settings.CARLA_HOSTS_TYPE == 'local' else '') + ' CARLA SIMULATOR',
    7: 'CARLA SIMULATOR RESTARTED',
}

