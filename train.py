from collections import deque
import os
import time
import settings
from threading import Thread
from Dependencies import CarlaEnv, CarlaEnvSettings, start_carla, get_hparams, TRAINER_STATE, check_weights_size, run_trainer, run_agent, ConsoleStats, Commands, STOP, AGENT_STATE, kill_carla, restart_carla, CARLA_SETTINGS_STATE, set_carla_sync_mode, Synchronizer
import settings 
from multiprocessing import Process, Value, Array, Queue, Condition



# training method, sets up everything
def train():

    print("starting...")

    start_time = time.time()

    # Create required folders
    os.makedirs('models', exist_ok=True) 
    os.makedirs('tmp', exist_ok=True) 
    os.makedirs('checkpoint', exist_ok=True)
    
    # starting the carla server instance
    if start_carla() == -1:
        return -1


    # load hparams
    hparams = get_hparams()
    if hparams:
        # If everything is ok, update start time by previous running time
        print('hparams found')
        if hparams['rl_algorithm'] != settings.RL_ALGORITHM:
            print("mismatch of action types")
            return -1
        
        start_time -= hparams['duration']
    rl_algorithm = hparams['rl_algorithm'] if hparams else settings.RL_ALGORITHM

    # Spawn limited trainer process and get weights' size
    weights_size = Value('L', 0)
    p = Process(target=check_weights_size, args=(hparams['model_path'] if hparams else False, rl_algorithm, weights_size), daemon=True)
    p.start()


    while weights_size.value != 0:
        time.sleep(0.01)
    p.join()

    # A bunch of variabled and shared variables used to set all parts of the training
    duration = Value('d')
    episode = Value('L', hparams['episode'] if hparams else 0)
    epsilon = Array('d', hparams['epsilon'] if hparams else [settings.START_EPSILON, settings.EPSILON_DECAY, settings.MIN_EPSILON])
    discount = Value('d', hparams['discount'] if hparams else settings.DISCOUNT)
    update_target_every = Value('L', hparams['update_target_every'] if hparams else settings.UPDATE_TARGET_EVERY)
    last_target_update = hparams['last_target_update'] if hparams else 0
    min_reward = Value('f', hparams['min_reward'] if hparams else settings.MIN_REWARD)
    agent_show_preview = []
    for agent in range(settings.AGENTS):
        if hparams and agent < len(hparams['agent_show_preview']):
            agent_show_preview.append(Array('f', hparams['agent_show_preview'][agent]))
        else:
            agent_show_preview.append(Array('f', [(agent + 1) in settings.AGENT_SHOW_PREVIEW, 0, 0, 0, 0, 0]))
    

    weights = Array('c', weights_size.value)

    sync_mode = hparams['sync_mode'] if hparams else settings.SYNC_MODE
    save_checkpoint_every = Value('L', hparams['save_checkpoint_every'] if hparams else settings.SAVE_CHECKPOINT_EVERY)
    seconds_per_episode = Value('L', hparams['seconds_per_episode'] if not sync_mode and hparams else settings.SECONDS_PER_EPISODE)
    steps_per_episode = Value('L', hparams['steps_per_episode'] if sync_mode and hparams else settings.STEPS_PER_EPISODE)
    
    weights_iteration = Value('L', hparams['weights_iteration'] if hparams else 0)
    transitions = Queue()
    tensorboard_stats = Queue()
    trainer_stats = Array('f', [0, 0])
    carla_check = None
    episode_stats = Array('d', [-10**6, -10**6, -10**6, 0, 0, 0, 0, -10**6, -10**6, -10**6] + [-10**6 for _ in range((CarlaEnv.action_space_size + 1) * 3)])
    stop = Value('B', 0)
    agent_stats = []
    for _ in range(settings.AGENTS):
        agent_stats.append(Array('f', [0, 0, 0]))
    optimizer = Array('d', [-1, -1, 0, 0, 0, 0]) if rl_algorithm == 'discrete' else Array('d', [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0])
    car_npcs = Array('L', hparams['car_npcs'] if hparams else [settings.CAR_NPCS, settings.RESET_CAR_NPC_EVERY_N_TICKS])
    pause_agents = []
    for _ in range(settings.AGENTS):
        pause_agents.append(Value('B', 0))
    print_console = Value('B', settings.PRINT_CONSOLE)

    # Run Carla settings (weather, NPC control) in a separate thread
    carla_settings_threads = [] 
    carla_settings_stats = []
    carla_frametimes_list = []
    carla_fps_counters = []
    carla_fps = []
    agents_in_carla_instance = {}
    fixed_delta_sec = hparams['fixed_delta_sec'] if hparams else settings.FIXED_DELTA_SEC if sync_mode else -1
    use_n_future_states = hparams['use_n_future_states'] if hparams else settings.USE_N_FUTURE_STEPS
    put_trans_every = hparams['put_trans_every'] if hparams else settings.PUT_TRANS_EVERY

    print("env")
    env_settings_cond = Condition() if sync_mode else None
    step_cond = Condition() if sync_mode else None
    synchronizer = Synchronizer() if sync_mode and settings.AGENTS > 1 else None
        
    for process_no in range(settings.CARLA_HOSTS_NO):
        agents_in_carla_instance[process_no] = []
        if sync_mode:
            set_carla_sync_mode(settings.CARLA_HOSTS[process_no], fixed_delta_sec)
    for agent in range(settings.AGENTS):
        carla_instance = 1 if not len(settings.AGENT_CARLA_INSTANCE) or settings.AGENT_CARLA_INSTANCE[agent] > settings.CARLA_HOSTS_NO else settings.AGENT_CARLA_INSTANCE[agent]
        agents_in_carla_instance[carla_instance-1].append(pause_agents[agent])
    for process_no in range(settings.CARLA_HOSTS_NO):
        carla_settings_process_stats = Array('f', [-1, -1, -1, -1, -1, -1])
        carla_frametimes = Queue()
        carla_frametimes_list.append(carla_frametimes)
        carla_fps_counter = deque(maxlen=60)
        carla_fps.append(Value('f', 0))
        carla_fps_counters.append(carla_fps_counter)
        carla_settings_stats.append(carla_settings_process_stats)
        carla_settings = CarlaEnvSettings(process_no, agents_in_carla_instance[process_no], sync_mode, env_settings_cond, step_cond, stop, car_npcs, carla_settings_process_stats)
        carla_settings_thread = Thread(target=carla_settings.update_settings_in_loop, daemon=True)
        carla_settings_thread.start()
        carla_settings_threads.append([carla_settings_thread, carla_settings])
        
    print("starting Trainer...")
    # Start trainer process
    trainer_process = Process(target=run_trainer, args=(hparams['model_path'] if hparams else False, hparams['logdir'] if hparams else False, stop, weights, weights_iteration, episode, rl_algorithm, hparams['trainer_iteration'] if hparams else 1, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, steps_per_episode, duration, transitions, tensorboard_stats, trainer_stats, episode_stats, optimizer, hparams['models'] if hparams else [], car_npcs, carla_settings_stats, carla_fps, sync_mode, fixed_delta_sec, use_n_future_states, put_trans_every), daemon=True)
    trainer_process.start()

    # Wait for trainer to be ready, it needs to, for example, dump weights that agents are going to update
    while trainer_stats[0] != TRAINER_STATE.waiting:
        time.sleep(0.1)
    
    
    print("starting Agent...")
    agents = []
    # start agents, can be more than 1
    for agent in range(settings.AGENTS):
        carla_instance = 1 if not len(settings.AGENT_CARLA_INSTANCE) or settings.AGENT_CARLA_INSTANCE[agent] > settings.CARLA_HOSTS_NO else settings.AGENT_CARLA_INSTANCE[agent]
        p = Process(target=run_agent, args=(agent, carla_instance-1, stop, pause_agents[agent], episode, rl_algorithm, epsilon, agent_show_preview[agent], weights, weights_iteration, transitions, tensorboard_stats, agent_stats[agent], carla_frametimes_list[carla_instance-1], seconds_per_episode, steps_per_episode, sync_mode, synchronizer, env_settings_cond, step_cond, use_n_future_states, put_trans_every), daemon=True)
        p.start()
        agents.append(p)



    print('\n'*(10))
    # Console stats to keep get the current state of the env and agents
    console_stats = ConsoleStats(print_console, stop, duration, start_time, episode, epsilon, trainer_stats, agent_stats, episode_stats, carla_fps, weights_iteration, optimizer, carla_settings_threads, seconds_per_episode)
    console_stats_thread = Thread(target=console_stats.print, daemon=True)
    console_stats_thread.start()
    
    

    commands = Commands(print_console, stop, epsilon, discount, update_target_every, min_reward, save_checkpoint_every, seconds_per_episode, steps_per_episode, agent_show_preview, optimizer, car_npcs, sync_mode)


    # Main loops
    while True:
        
        # If everything is running or carla broke...
        if stop.value in[STOP.running, STOP.carla_simulator_error, STOP.restarting_carla_simulator, STOP.carla_simulator_restarted]:

            
            # ...and all agents return an error
            if any([state[0] == AGENT_STATE.error for state in agent_stats]):
                # If it's a running state, set it to carla error
                if stop.value == STOP.running:
                    stop.value = STOP.carla_simulator_error
                    for process_no in range(settings.CARLA_HOSTS_NO):
                        carla_fps_counters[process_no].clear()

            # If agents are not returning errors, set running state
            else:
                stop.value = STOP.running
                carla_check = None

        # Append new frametimes from carla for stats
        if not stop.value == STOP.carla_simulator_error:
            for process_no in range(settings.CARLA_HOSTS_NO):
                for _ in range(carla_frametimes_list[process_no].qsize()):
                    try:
                        carla_fps_counters[process_no].append(carla_frametimes_list[process_no].get(True, 0.1))
                    except:
                        break
                carla_fps[process_no].value = len(carla_fps_counters[process_no]) / sum(carla_fps_counters[process_no]) if sum(carla_fps_counters[process_no]) > 0 else 0
        
        # If carla broke
        if stop.value == STOP.carla_simulator_error and settings.CARLA_HOSTS_TYPE == 'local':
            # First check, set a timer because...
            if carla_check is None:
                carla_check = time.time()
            # ...we give it 15 seconds to possibly recover, if not...
            if time.time() > carla_check + 15:
                
                # ... set Carla restart state and try to restart it
                stop.value = STOP.restarting_carla_simulator
                if settings.CARLA_HOSTS_TYPE == 'local':
                    kill_carla()
                for process_no in range(settings.CARLA_HOSTS_NO):
                    carla_settings_threads[process_no][1].clean_car_npcs()
                    carla_settings_threads[process_no][1].restart = True
                    carla_fps_counters[process_no].clear()
                    carla_fps[process_no].value = 0
                for process_no in range(settings.CARLA_HOSTS_NO):
                    while not carla_settings_threads[process_no][1].state == CARLA_SETTINGS_STATE.restarting:
                        time.sleep(0.1)
                restart_carla()
                for process_no in range(settings.CARLA_HOSTS_NO):
                    carla_settings_threads[process_no][1].restart = False
                stop.value = STOP.carla_simulator_restarted

        # When Carla restarts, give it up to 60 seconds, then try again if failed
        if stop.value == STOP.restarting_carla_simulator and time.time() > carla_check + 60:
            stop.value = STOP.carla_simulator_error
            carla_check = time.time() - 15

        # Process commands
        commands.process()

        # If stopping - cleanup and exit
        if stop.value == STOP.stopping:

            # Trainer process already "knows" that, just wait for it to exit
            trainer_process.join()

            # The same for all agents
            for agent in agents:

                agent.join()

            # ... and Carla settings
            for process_no in range(settings.CARLA_HOSTS_NO):
                carla_settings_threads[process_no][0].join()

            # Close Carla
            kill_carla()

            stop.value = STOP.stopped
            time.sleep(1)
            break

        time.sleep(0.01)
        
if __name__ == '__main__':
    train()



