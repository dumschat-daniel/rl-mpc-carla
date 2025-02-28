
import os
import settings
from threading import Thread
from Dependencies import CarlaEnvSettings, start_carla, get_hparams, test_agent, ConsoleStats, set_carla_sync_mode
import settings 
from multiprocessing import Condition
import argparse
import re

# Replacement for shared multiprocessing object when testing
class PauseObject:
    value: int = 0

def test(mpc_flag, model_path):
    hparams = None
    if mpc_flag:
        # model none means we use mpc for testing
        model = None
    elif model_path:
        model = model_path
    else:
        # Load hparams if they are being saved by trainer
        hparams = get_hparams()
        if hparams is None:
            print("No hparams file found. Either place models and a correct hparams file in the checkpoint directory, or apply the --mpc or --path flag to use MPC or your own model.")
            return -1

        print('Saved models:')

        # Check paths from hparams, filter out models whose files exist on disk
        filtered_models = {}
        for model in hparams['models']:
            # Add algorithm suffix if needed
            if settings.TESTING_RL_ALGORITHM != 'dqn':
                model = model + f'_{settings.TESTING_RL_ALGORITHM}_actor.model'
            
            # Ensure the file exists
            if os.path.isdir(model):
                # Extract filename from full path
                filename = os.path.basename(model)

                # Use regex to extract the last number before .model
                match = re.search(r'_([\d]+)(?:_[^_]+)*\.model$', filename)

                if match:
                    number = int(match.group(1))
                    filtered_models[number] = model
                else:
                    print("Unknown model: ", filename)  # Debug print for unmatched files


        # Sort models by extracted number
        filtered_models = list(dict(sorted(filtered_models.items(), key=lambda kv: kv[0])).values())

        # Display sorted models
        for index, model in enumerate(filtered_models):
            print(f'{index + 1}. {model}')

        # Ask user to choose a model
        model_index = input('Choose the model (empty for most recent one): ')
        if model_index:
            model = filtered_models[int(model_index) - 1]
        else:
            model = filtered_models[-1]  # Most recent model if no input provided

        if model is not None:
            pattern = r'(_td3_actor\.model|_ddpg_actor\.model|_dqn\.model)$'
            model = re.sub(pattern, '', model)
    print('Starting...')

    # Kill Carla processes if there are any and start simulator
    start_carla(testing=True)

    sync_mode = hparams['sync_mode'] if hparams else settings.SYNC_MODE
    fixed_delta_sec = hparams['fixed_delta_sec'] if hparams else settings.FIXED_DELTA_SEC if sync_mode else -1
    if sync_mode:
        set_carla_sync_mode(settings.TESTING_CARLA_HOST[0], fixed_delta_sec)
        
    env_settings_cond = Condition() if sync_mode else None
    step_cond = Condition() if sync_mode else None
    
    car_npcs = hparams['car_npcs'] if hparams else [settings.CAR_NPCS, settings.RESET_CAR_NPC_EVERY_N_TICKS]
    # Run Carla settings (weather, NPC control) in a separate thread
    pause_object = PauseObject()
    carla_settings = CarlaEnvSettings(0, [pause_object], sync_mode, env_settings_cond, step_cond, car_npcs=car_npcs)
    carla_settings_thread = Thread(target=carla_settings.update_settings_in_loop, daemon=True)
    carla_settings_thread.start()
    
    print(f'Starting agent ({model})...')
    test_agent(model, sync_mode, env_settings_cond, step_cond, pause_object, None, ConsoleStats.print_short)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="start testing with mpc or a provided model")
    
    parser.add_argument('--mpc', action='store_true', help='runs the tests with mpc directly')
    
    parser.add_argument('--model', type=str, help='Provide model path (actor)')

    args = parser.parse_args()

    test(args.mpc, args.model)