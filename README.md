# Bachelor Thesis Code Repository

This repository contains the code associated with the bachelor thesis of **Daniel Dumschat** titled:
**"Verwendung von Model Predictive Control zur Effizienzsteigerung und Stabilisierung von Reinforcement Learning Algorithmen für autonomes Fahren"**  (*"Use of Model Predictive Control to Enhance Efficiency and Stabilize Reinforcement Learning Algorithms for Autonomous Driving"*).

---
## Setup and Installation
### 1. CARLA Server

- **Clone the CARLA Server:**  Visit the [CARLA Releases page](https://github.com/carla-simulator/carla/releases) and download the appropriate version.
- **Version Note:** This code has been created and tested on **CARLA Version 0.9.15** which requires **Python Version 3.7**.
### 2. Clone This Repository
- **Repository Placement:**  Clone this repository and place it in the `PythonAPI` folder of the CARLA installation. Alternatively, if you place it elsewhere, make sure to update the `CARLA_PATH` in `settings.py`.

```bash
git clone https://git.thm.de/ddms28/bachelor-rl-mpc-carla
```
1. Optional: Anaconda Environment
Anaconda Setup: It is recommended to create an Anaconda environment for dependency management.
    Download and install Anaconda if you haven’t already: https://www.anaconda.com/download
2. Install Python Dependencies
Using pip, install the required Python packages with the following command:
```bash
pip install -r requirements.txt
```


### 3. Quick Start
- `settings.py` contains settings for both training and testing.

#### Testing
- There are already trained model in `example_models`, to load a model just place the models and the hparams file in the checkpoint directory.
- To load a model directly or run the tests using mpc you can add the --model "path" or --mpc flag.
- This will run the model through the specified testing scenarios.

#### Training
- To train just run `train.py`. This will check the checkpoint directory for a model save to continue training from.
- To train from start either set USE_HPARAMS to false or make sure the checkpoint directory is empty. 
- To load a different Model than the newest, configure the model parameter in the hparams file.
- Either setup training with default settings. For that just set the first parameter in `settings.py` "QUICK_SETUP" to either one of the phases (1,2,3) or to a default algorithm (dqn, ddpg, td3) and just run the training. Otherwise configure settings to change the Simulation and Training.


---


### **Modifying the Simulation**
- All modifiable parameters can be changed using the `settings.py` file. This file contains settings for training, testing, and CARLA configurations.
- The most important settings are:
    - **CARLA_PATH**: This is the path to the CARLA installation and needs to be adjusted if this repository is not placed in the `PythonAPI` folder.
    - **CARLA_HOSTS**: Contains a list of hosts and sets up the servers.
    - **SETUP_TIME**: The time it takes for a spawned vehicle to be ready to accept inputs. 0.7 should be enough. Can be teted in the helper class.
    - **SYNC_MODE**: Should always be set to `True`. This is required for MPC, and asynchronous mode is neither tested nor recommended for training.
    - **AGENTS**: The number of agents used simultaneously for exploration.
    - **REWARD settings**: These include several parameters, with `reward_function_metrics` containing all enabled reward factors. These can be adjusted as needed.
    - **USE_HPARAMS**: Whether to continue training from the most recent saved state.
    - **SAVE_CHECKPOINT_EVERY**: Defines how often (in training episodes) the model weights are saved.
    - **RL_ALGORITHM**: The algorithm used for training. _Note_: DQN is not compatible with MPC.
    - **NOISE_TYPE**: Specifies either `GaussianNoise` or `OUNoise`, particularly relevant for DDPG and TD3 strategies.
    - **Algorithm-specific settings**: Parameters tailored to the chosen algorithm.
    - **USE_MPC and MPC_Phase**: Specifies whether to use MPC during training and in which phase, determining how MPC is integrated into the training process.
    - **Sensor settings**: Individual configurations for each sensor.
    - and more 
- The `helpers` directory contains small programs that provide additional information regarding some settings.


### **Structure of the Repository**

- `train.py` and `test.py` are the starting points for training and testing
- `Dependencies` contains the main code functionality 
- `checkpoint` is used for saving and loading model states
- `tmp` is used to influence the Training
- `helpers` contains small pieces of code that help setting up the Code
- `models` contains saved models based on metrics
- `example_models` contains some models from training that can be trained further or used for testing


### **Influencing the Training**
- During training, some parameters can be modified or the process influenced in general.
- This is done through command files (text files with `command_` as a prefix). These files should be placed in the `tmp` folder. Some relevant commands are:
    - `checkpoint save_every`: Changes how frequently the model is saved.
    - `preview 'agent_nr' on/off`: Enables or disables a preview of the specified agent.
    - `stop now/checkpoint`: Stops training immediately or at the next checkpoint, saving the model at that point.
- For more details, refer to the `commands.py` file.