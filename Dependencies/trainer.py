from collections import deque
from dataclasses import dataclass
import random
from threading import Thread

import numpy as np
import settings
import os
import json
from Dependencies import ARTDQNAgent, TensorB, STOP, ACTIONS, ACTIONS_NAMES, NeptuneLogger, Reward_priorisation_experience_replay, PrioritizedReplayMemory
import time
import pickle
import logging
import shutil
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
tf.get_logger().setLevel(logging.ERROR)



def get_hparams():
    """Load Hparams Object to continue training / test."""
    hparams = None
    if os.path.isfile('checkpoint/hparams.json') and settings.USE_HPARAMS:
        with open('checkpoint/hparams.json', encoding='utf-8') as f:
            hparams = json.load(f)
    return hparams


def check_weights_size(model_path, rl_algorithm, weights_size):
    """create a model and save serialized weights' size."""
    trainer = ARTDQNTrainer(model_path, rl_algorithm)
    weights_size.value = len(trainer.serialize_weights())   



class ARTDQNTrainer(ARTDQNAgent):
    """Trainer Class inherits from Agent so that it can use its functionality. Responsible training, updating weights, sharing them with the agent and logging."""
    def __init__(self, model_path, rl_algorithm):
        self.model_path = model_path
        self.rl_algorithm = rl_algorithm
        self.show_conv_cam = False

        # creates the model used for prediction
        if self.rl_algorithm == 'dqn':
            self.model = self.create_model()
        else:
            self.actor_model = self.create_model(prediction=True)
        
    def init_training(self, stop, logdir, trainer_stats, episode, trainer_iteration, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, steps_per_episode, duration, optimizer, models, car_npcs, sync_mode, fixed_delta_seconds, use_n_future_states, put_trans_every):
        """Sets up Training."""
        self.show_conv_cam = False
        if settings.EXPERIENCE_REPLAY_METHOD == 'reward_old':
            self.experience_replay = Reward_priorisation_experience_replay()
        else:
            self.experience_replay = PrioritizedReplayMemory(settings.EXPERIENCE_REPLAY_SIZE, settings.EXPERIENCE_REPLAY_ALPHA, settings.EXPERIENCE_REPLAY_BETA, settings.EXPERIENCE_REPLAY_LAMBDA, settings.EXPERIENCE_REPLAY_METHOD)
        self.trainer_iteration = trainer_iteration
        self.logdir = logdir if logdir else "logs/{}-{}".format(settings.MODEL_NAME, int(time.time()))
        self.tensorboard = TensorB(log_dir=self.logdir)
        self.tensorboard.step = episode.value

        
        self.neptune_logger = NeptuneLogger() if 'neptune' in settings.ADDITIONAL_LOGGING else None
        if self.neptune_logger:
            self.neptune_logger.step = episode.value if 'neptune' in settings.ADDITIONAL_LOGGING else None

       
        self.last_target_update = last_target_update

        # Internal properties
        self.last_log_episode = 0
        self.tps = 0
        self.last_checkpoint = 0
        self.save_model = False

        # Shared properties - either used by model or only for checkpoint purposes
        self.stop = stop
        self.trainer_stats = trainer_stats
        self.episode = episode
        self.epsilon = epsilon
        self.discount = discount
        self.update_target_every = update_target_every
        self.min_reward = min_reward
        self.agent_show_preview = agent_show_preview
        self.save_checkpoint_every = save_checkpoint_every
        self.seconds_per_episode = seconds_per_episode
        self.steps_per_episode = steps_per_episode
        self.duration = duration
        self.optimizer = optimizer
        self.models = models
        self.car_npcs = car_npcs


        self.sync_mode = sync_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        
        self.use_n_future_steps = use_n_future_states
        self.put_trans_every = put_trans_every
        
        if self.rl_algorithm == 'dqn':
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

            self.optimizer[0], self.optimizer[1] = self.get_lr_decay('model')

        else:
             
            # lr schedule only used if settings.DDPG_LR_DECAY is True
            actor_lr_schedule = ExponentialDecay(
                initial_learning_rate=settings.DDPG_ACTOR_OPTIMIZER_LEARNING_RATE,
                decay_steps=5000,
                decay_rate=settings.DDPG_ACTOR_OPTIMIZER_DECAY,
                staircase=True  # For discrete decay steps; set to False for smooth decay
            )

            critic_lr_schedule = ExponentialDecay(
                initial_learning_rate=settings.DDPG_CRITIC_OPTIMIZER_LEARNING_RATE,
                decay_steps=5000,
                decay_rate=settings.DDPG_CRITIC_OPTIMIZER_DECAY,
                staircase=True
            )

            self.mpc_critic_epsilon = settings.MPC_CRITIC_START_EPSILON
            if self.rl_algorithm == 'ddpg':

                self.critic_model= self.create_model()

                self.target_actor_model = self.create_model(prediction=True) 
                self.target_actor_model.set_weights(self.actor_model.get_weights())
                
                self.target_critic_model = self.create_model()
                self.target_critic_model.set_weights(self.critic_model.get_weights())
                
                self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr_schedule if settings.DDPG_LR_DECAY else settings.DDPG_ACTOR_OPTIMIZER_LEARNING_RATE)
                self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr_schedule if settings.DDPG_LR_DECAY else settings.DDPG_CRITIC_OPTIMIZER_LEARNING_RATE)

            elif self.rl_algorithm == 'td3':

                self.critic_model1 = self.create_model(critic_number=1)
                self.critic_model2 = self.create_model(critic_number=2)

                self.target_actor_model = self.create_model(prediction=True) 
                self.target_actor_model.set_weights(self.actor_model.get_weights())

                self.target_critic_model1 = self.create_model()
                self.target_critic_model1.set_weights(self.critic_model1.get_weights())

                self.target_critic_model2 = self.create_model()
                self.target_critic_model2.set_weights(self.critic_model2.get_weights())

                self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr_schedule if settings.DDPG_LR_DECAY else settings.DDPG_ACTOR_OPTIMIZER_LEARNING_RATE)
                self.critic_optimizer1 = tf.keras.optimizers.Adam(learning_rate=critic_lr_schedule if settings.DDPG_LR_DECAY else settings.DDPG_CRITIC_OPTIMIZER_LEARNING_RATE)
                self.critic_optimizer2 = tf.keras.optimizers.Adam(learning_rate=critic_lr_schedule if settings.DDPG_LR_DECAY else settings.DDPG_CRITIC_OPTIMIZER_LEARNING_RATE)
        
    
    
    def compute_td_error(self, transition):
        states, action, reward, done, _ = transition
        try:
            state = [
                tf.convert_to_tensor(states[0][0], dtype=tf.float32),  # Navigation input (1,2)
                tf.convert_to_tensor(states[0][1], dtype=tf.float32),  # Speed input (1,1)
                tf.convert_to_tensor(states[0][2], dtype=tf.float32)   # Lane center input (1,2)
            ]

            next_state = [
                tf.convert_to_tensor(states[1][0], dtype=tf.float32), 
                tf.convert_to_tensor(states[1][1], dtype=tf.float32), 
                tf.convert_to_tensor(states[1][2], dtype=tf.float32)   
            ]

            next_action = self.target_actor_model(next_state, training=False)

            next_action = tf.reshape(next_action, (1, next_action.shape[-1]))

            action = tf.convert_to_tensor(action, dtype=tf.float32)
            action = tf.reshape(action, (1, action.shape[-1]))

            target_q = self.target_critic_model([next_state, next_action], training=False)

            td_target = reward + (1.0 - done) * settings.DISCOUNT * target_q

            current_q = self.critic_model([state, action], training=False)

            td_error = tf.abs(td_target - current_q)

            return td_error.numpy()

        except Exception as e:
            print(f"Error in compute_td_error: {e}")
            return None


    def update_experience_replay(self, transition):
        """Adds Transition to replay memory based on the experience replay method."""
        if settings.EXPERIENCE_REPLAY_METHOD == 'reward_old':
            self.experience_replay.add(transition)
        else:
            td_error = self.compute_td_error(transition)
            #td_error = 0
            if td_error is None:
                print("Skipping transition due to TD error computation failure.")
                return
                
            
            formatted_transition = (transition, td_error, 0.0)

            self.experience_replay.add(formatted_transition)

            
        
    
    def get_lr_decay(self, model_type):
        """lr decay only for dqn."""
        if model_type == 'model':
            model = self.model

        lr = tf.convert_to_tensor(model.optimizer.lr)
        initial_decay = tf.convert_to_tensor(getattr(model.optimizer, 'initial_decay', 0.0))
        decay = tf.convert_to_tensor(getattr(model.optimizer, 'decay', 0.0))  # or 'learning_rate_decay' if applicable
        if initial_decay > 0 or decay > 0:
            lr = lr * (1. / (1. + decay * tf.cast(model.optimizer.iterations, tf.float32)))
        return lr.numpy(), decay.numpy()
    
    
    def serialize_weights(self):
        """serialized weights are being shared between the trainer and the agent"""
        if self.rl_algorithm == 'dqn':
            return pickle.dumps(self.model.get_weights())
        else: #both ddpg and td3
            return pickle.dumps(self.actor_model.get_weights())

    def init_serialized_weights(self, weights, weights_iteration):

        self.weights = weights
        self.weights.raw = self.serialize_weights()
        self.weights_iteration = weights_iteration 

        
    def soft_update(self, target_model, source_model):
        """update target networks based on tau."""
        tau = settings.TAU
        if tau is None:
            # Perform hard update if tau is None
            target_model.set_weights(source_model.get_weights())
        else:
            # Perform soft update 
            target_weights = target_model.get_weights()
            source_weights = source_model.get_weights()
            new_weights = [tau * source + (1 - tau) * target for target, source in zip(target_weights, source_weights)]
            target_model.set_weights(new_weights)



    '''
    All Training Function calls for dqn, ddpg, td3.
    '''

    def train_dqn(self, current_states_stacked, next_states_stacked, actions, rewards, dones, minibatch):
        current_qs = self.model.predict(current_states_stacked, settings.PREDICTION_BATCH_SIZE, verbose=0)
        next_qs = self.target_model.predict(next_states_stacked, settings.PREDICTION_BATCH_SIZE, verbose=0)
        target_qs = current_qs.copy()

        # Update target Q-values for DQN
        for i in range(len(minibatch)):
            if dones[i]:
                target_qs[i, actions[i]] = rewards[i]
            else:
                max_future_q = np.max(next_qs[i])
                target_qs[i, actions[i]] = rewards[i] + settings.DISCOUNT * max_future_q

        log_this_step = False
        if self.tensorboard.step > self.last_log_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # Train the DQN model
        history = self.model.fit(current_states_stacked, target_qs, batch_size=settings.TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        loss = history.history.get('loss', [None])[0]
        accuracy = history.history.get('accuracy', [None])[0]

        avg_q_value = np.mean(current_qs)
        max_q_value = np.max(current_qs)
        min_q_value = np.min(current_qs)

        q_values_exp = np.exp(current_qs - np.max(current_qs))  # subtract max for numerical stability
        probs = q_values_exp / np.sum(q_values_exp)

        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-10, None)))  # clip to avoid log(0)
        logs = {
            'step': self.trainer_iteration,
            'training_loss': loss,
            'training_accuracy': accuracy,
            'learning_rate': self.optimizer[3],
            'learning_rate_decay': self.optimizer[5],
            'batch_reward_avg': np.mean(rewards),
            'batch_reward_min': np.min(rewards),
            'batch_reward_max': np.max(rewards),
            'batch_size': len(minibatch),
            'avg_q_value': avg_q_value,
            'max_q_value': max_q_value,
            'min_q_value': min_q_value,
            'entropy': entropy
    }
        
        if self.trainer_iteration % settings.UPDATE_TARGET_EVERY == 0:
            self.target_model.set_weights(self.model.get_weights())

        if self.optimizer[2] == 1:
            self.optimizer[2] = 0
            self.compile_model(model=self.model, learning_rate=self.optimizer[3], decay=self.get_lr_decay('model')[1])
        if self.optimizer[4] == 1:
            self.optimizer[4] = 0
            self.compile_model(model=self.model, learning_rate=self.get_lr_decay('model')[0], decay=self.optimizer[5])
            self.optimizer[0], self.optimizer[1] = self.get_lr_decay('model')

        return logs


    def train_ddpg(self, current_states_stacked, next_states_stacked, actions, mpc_actions, rewards, dones, minibatch, indices=None, is_weights=None):
        """ Updates the DDPG agent by training both the critic and actor networks.
            Optionally integrates MPC loss depending on the training phase."""
        # set impotrance sampling weights to 1 (no adaption) if no importance sampling
        if is_weights is not None:
            is_weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        else:
            is_weights = tf.ones_like(rewards, dtype=tf.float32)

        # Critic
        target_actions = self.target_actor_model.predict(next_states_stacked, verbose=0)

        # scale actions (for full carla actions scaling is 1 to 1)
        if settings.SCALE_THROTTLE:
            scaled_throttle = (settings.SCALE_THROTTLE[1] - settings.SCALE_THROTTLE[0]) / 2 * target_actions[0][0] + \
                  (settings.SCALE_THROTTLE[1] + settings.SCALE_THROTTLE[0]) / 2
            target_actions[0][0] = scaled_throttle

        if settings.SCALE_STEER:
            scaled_steer = (settings.SCALE_STEER[1] - settings.SCALE_STEER[0]) / 2 * target_actions[0][1] + \
                  (settings.SCALE_STEER[1] + settings.SCALE_STEER[0]) / 2
            target_actions[0][1] = scaled_steer

        # Predict Q-values from target critic network
        predicted_q_values = self.target_critic_model.predict([next_states_stacked, target_actions], verbose=0)

        # Compute Bellman target values
        target_q_values = rewards + settings.DISCOUNT * (1 - dones) * tf.squeeze(predicted_q_values)

        with tf.GradientTape() as tape:

            q_values = tf.squeeze(self.critic_model([current_states_stacked, actions]), 1)
            td_errors = tf.abs(target_q_values - q_values)

            # combined loss function
            if settings.USE_MPC and settings.MPC_PHASE == 'Transition':
                # Calculate MPC loss
                mpc_loss = tf.reduce_mean(tf.square(actions - mpc_actions))
                # Blend MPC loss with RL critic loss using epsilon
                mpc_loss = tf.cast(mpc_loss, tf.float32)
                target_q_values = tf.cast(target_q_values, tf.float32)
                q_values = tf.cast(q_values, tf.float32)
            
                critic_loss = self.mpc_critic_epsilon * mpc_loss + (1 - self.mpc_critic_epsilon) * tf.reduce_mean(is_weights * tf.square(td_errors))
            else:
                # Standard critic loss
                critic_loss = tf.reduce_mean(is_weights * tf.square(td_errors))

        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        # gradient clipping
        if settings.CRITIC_GRADIENT_CLIP_NORM:
            critic_grads = [tf.clip_by_norm(grad, settings.CRITIC_GRADIENT_CLIP_NORM) for grad in critic_grads if grad is not None]
        
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))
        critic_grad_norms = [tf.norm(grad).numpy() for grad in critic_grads if grad is not None]

        critic_avg_q_value = np.mean(q_values)
        critic_max_q_value = np.max(q_values)
        critic_min_q_value = np.min(q_values)

        # ACTOR
        if settings.USE_MPC and settings.MPC_PHASE == 'Imitation':
            # Use MPC loss for the actor during the imitation phase
            with tf.GradientTape() as tape:
                predicted_actions = self.actor_model(current_states_stacked)
                mpc_loss = tf.reduce_mean(tf.square(predicted_actions - mpc_actions))  # MPC-guided loss
            
            actor_grads = tape.gradient(mpc_loss, self.actor_model.trainable_variables)
            actor_loss = mpc_loss 
        else:
            # Default actor loss
            with tf.GradientTape() as tape:
                predicted_actions = self.actor_model(current_states_stacked)
                q_values = self.critic_model([current_states_stacked, predicted_actions])
                actor_loss = -tf.reduce_mean(q_values)
            
            actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)

        # Gradient clipping
        if settings.ACTOR_GRADIENT_CLIP_NORM:
            actor_grads = [tf.clip_by_norm(grad, settings.ACTOR_GRADIENT_CLIP_NORM) for grad in actor_grads if grad is not None]
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        actor_grad_norms = [tf.norm(grad).numpy() for grad in actor_grads]
        
        # STATISTICS
        actor_avg_q_value = np.mean(q_values)
        actor_max_q_value = np.max(q_values)
        actor_min_q_value = np.min(q_values)

        action_distribution = np.histogram(actions, bins=10, range=(-1, 1))[0]
        entropy = -np.sum(action_distribution * np.log(action_distribution + 1e-10)) / len(actions)

        logs = {
        'step': self.trainer_iteration,
        'critic_loss': critic_loss.numpy(),
        'critic_gradient_mean': np.mean(critic_grad_norms),
        'critic_gradient_max': np.max(critic_grad_norms),
        'critic_gradient_min': np.min(critic_grad_norms),
        'actor_loss': actor_loss.numpy(),
        'actor_gradient_mean': np.mean(actor_grad_norms),
        'actor_gradient_max': np.max(actor_grad_norms),
        'actor_gradient_min': np.min(actor_grad_norms),
        'batch_reward_avg': np.mean(rewards),
        'batch_reward_min': np.min(rewards),
        'batch_reward_max': np.max(rewards),
        'batch_size': len(minibatch),
        'actor_avg_q_value': actor_avg_q_value,
        'actor_max_q_value': actor_max_q_value,
        'actor_min_q_value': actor_min_q_value,
        'critic_avg_q_value': critic_avg_q_value,
        'critic_max_q_value': critic_max_q_value,
        'critic_min_q_value': critic_min_q_value,
        'entropy': entropy,
        'target_q_avg': np.mean(target_q_values),
        'target_q_max': np.max(target_q_values),
        'target_q_min': np.min(target_q_values),
        'mpc_critic_epsilon': self.mpc_critic_epsilon
    }
        
        # decay critic epsilon in Transition phase
        if settings.MPC_PHASE == 'Transition':
            self.mpc_critic_epsilon = max(settings.MPC_CRITIC_EPSILON_MIN, self.mpc_critic_epsilon * settings.MPC_CRITIC_EPSILON_DECAY)
    
        # update target Network
        if self.trainer_iteration % settings.UPDATE_TARGET_EVERY == 0:
            self.soft_update(self.target_actor_model, self.actor_model)
            self.soft_update(self.target_critic_model, self.critic_model)

        # update experience replay priorities
        if indices is not None:
            self.experience_replay.update_priorities(indices, td_errors.numpy())

        if self.optimizer[2] == 1:
            self.optimizer[2] = 0
            #self.compile_model(model=self.critic_model, learning_rate=self.optimizer[3], decay=self.get_lr_decay('critic')[1])
            self.critic_optimizer.learning_rate.assign(self.optimizer[3])
        if self.optimizer[4] == 1:
            self.optimizer[4] = 0
            # implement decay if needed
            #self.compile_model(model=self.critic_model, learning_rate=self.get_lr_decay('critic')[0], decay=self.optimizer[5])
            #self.optimizer[0], self.optimizer[1] = self.get_lr_decay('critic')

        if self.optimizer[8] == 1:
            self.optimizer[8] = 0
            self.actor_optimizer.learning_rate.assign(self.optimizer[9])
        if self.optimizer[10] == 1:
            self.optimizer[10] = 0
            # implement decay if needed

        return logs


    def train_td3(self, current_states_stacked, next_states_stacked, actions, mpc_actions, rewards, dones, minibatch, indices=None, is_weights=None):
        # set importance sampling weights to 1 (no adaption) if no importance sampling
        if is_weights is not None:
            is_weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        else:
            is_weights = tf.ones_like(rewards, dtype=tf.float32)
            
        # CRITIC
        target_actions = self.target_actor_model.predict(next_states_stacked, verbose=0)
        
        smoothing_noise = np.random.normal(0, settings.TD3_SMOOTHING_STD, size=target_actions.shape)
        smoothing_noise = np.clip(smoothing_noise, -settings.TD3_NOISE_CLIP, settings.TD3_NOISE_CLIP)

        # Scale target actions BEFORE adding noise
        if settings.SCALE_THROTTLE:
            target_actions[:, 0] = (settings.SCALE_THROTTLE[1] - settings.SCALE_THROTTLE[0]) / 2 * target_actions[:, 0] + \
                                (settings.SCALE_THROTTLE[1] + settings.SCALE_THROTTLE[0]) / 2

        if settings.SCALE_STEER:
            target_actions[:, 1] = (settings.SCALE_STEER[1] - settings.SCALE_STEER[0]) / 2 * target_actions[:, 1] + \
                                (settings.SCALE_STEER[1] + settings.SCALE_STEER[0]) / 2

        # Add noise and clip within scaled throttle and steer bounds
        target_actions[:, 0] = np.clip(target_actions[:, 0] + smoothing_noise[:, 0],
                                    settings.SCALE_THROTTLE[0], settings.SCALE_THROTTLE[1])
        target_actions[:, 1] = np.clip(target_actions[:, 1] + smoothing_noise[:, 1],
                                    settings.SCALE_STEER[0], settings.SCALE_STEER[1])
    

        target_q1 = self.target_critic_model1.predict([next_states_stacked, target_actions], verbose=0)
        target_q2 = self.target_critic_model2.predict([next_states_stacked, target_actions], verbose=0)
        target_q_values = rewards + settings.DISCOUNT * (1 - dones) * np.minimum(target_q1, target_q2).squeeze()

        

        # same as ddpg but with 2 critics
        with tf.GradientTape() as tape1:
            q1_values = tf.squeeze(self.critic_model1([current_states_stacked, actions]), 1)

            td_errors = tf.abs(target_q_values - q1_values)
            if settings.USE_MPC and settings.MPC_PHASE == 'Transition':
                mpc_loss = tf.reduce_mean(tf.square(actions - mpc_actions))
                critic_loss1 = self.mpc_critic_epsilon * mpc_loss + (1 - self.mpc_critic_epsilon) * tf.reduce_mean(is_weights * tf.square(td_errors))
            else:
                critic_loss1 = tf.reduce_mean(is_weights * tf.square(td_errors))
        critic_grads1 = tape1.gradient(critic_loss1, self.critic_model1.trainable_variables)
        if settings.CRITIC_GRADIENT_CLIP_NORM:
            critic_grads1 = [tf.clip_by_norm(grad, settings.CRITIC_GRADIENT_CLIP_NORM) for grad in critic_grads1 if grad is not None]
        self.critic_optimizer1.apply_gradients(zip(critic_grads1, self.critic_model1.trainable_variables))

        critic1_grad_norms = [tf.norm(grad).numpy() for grad in critic_grads1 if grad is not None]

        with tf.GradientTape() as tape2:
            q2_values = tf.squeeze(self.critic_model2([current_states_stacked, actions]), 1)

            td_errors = tf.abs(target_q_values - q2_values)
            if settings.USE_MPC and settings.MPC_PHASE == 'Transition':
                mpc_loss = tf.reduce_mean(tf.square(actions - mpc_actions))
                critic_loss2 = self.mpc_critic_epsilon * mpc_loss + (1 - self.mpc_critic_epsilon) * tf.reduce_mean(is_weights * tf.square(td_errors))
            else:
                critic_loss2 = tf.reduce_mean(is_weights * tf.square(td_errors))
        critic_grads2 = tape2.gradient(critic_loss2, self.critic_model2.trainable_variables)
        if settings.CRITIC_GRADIENT_CLIP_NORM:
            critic_grads2 = [tf.clip_by_norm(grad, settings.CRITIC_GRADIENT_CLIP_NORM) for grad in critic_grads2 if grad is not None]
        self.critic_optimizer2.apply_gradients(zip(critic_grads2, self.critic_model2.trainable_variables))

        critic2_grad_norms = [tf.norm(grad).numpy() for grad in critic_grads2 if grad is not None]

        actor_grad_norms = None
        if self.trainer_iteration % settings.TD3_DELAYED_POLICY_UPDATE == 0:
            
            # ACTOR
            if settings.USE_MPC and settings.MPC_PHASE == 'Imitation':
                # Use MPC loss for the actor during the imitation phase
                with tf.GradientTape() as tape:
                    predicted_actions = self.actor_model(current_states_stacked)
                    mpc_loss = tf.reduce_mean(tf.square(predicted_actions - mpc_actions)) 
                
                actor_grads = tape.gradient(mpc_loss, self.actor_model.trainable_variables)
                actor_loss = mpc_loss  
            else:
                # Default actor loss
                with tf.GradientTape() as tape:
                    predicted_actions = self.actor_model(current_states_stacked)
                    q_values = self.critic_model1([current_states_stacked, predicted_actions])
                    actor_loss = -tf.reduce_mean(q_values)
                
                actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        
            if settings.ACTOR_GRADIENT_CLIP_NORM:
                actor_grads = [tf.clip_by_norm(grad, settings.ACTOR_GRADIENT_CLIP_NORM) for grad in actor_grads if grad is not None]
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

            actor_grad_norms = [tf.norm(grad).numpy() for grad in actor_grads if grad is not None]
            # Soft-update target networks
            self.soft_update(self.target_actor_model, self.actor_model)
            self.soft_update(self.target_critic_model1, self.critic_model1)
            self.soft_update(self.target_critic_model2, self.critic_model2)

        action_distribution = np.histogram(actions, bins=10, range=(-1, 1))[0]
        entropy = -np.sum(action_distribution * np.log(action_distribution + 1e-10)) / len(actions)

        # STATISTICS
        logs = {
            'step': self.trainer_iteration,
            'critic_loss1': critic_loss1.numpy(),
            'critic_loss2': critic_loss2.numpy(),
            'critic_gradient_mean1': np.mean(critic1_grad_norms),
            'critic_gradient_max1': np.max(critic1_grad_norms),
            'critic_gradient_min1': np.min(critic1_grad_norms),
            'critic_gradient_mean2': np.mean(critic2_grad_norms),
            'critic_gradient_max2': np.max(critic2_grad_norms),
            'critic_gradient_min2': np.min(critic2_grad_norms),
            'avg_q1_value': np.mean(q1_values),
            'max_q1_value': np.max(q1_values),
            'min_q1_value': np.min(q1_values),
            'avg_q2_value': np.mean(q2_values),
            'max_q2_value': np.max(q2_values),
            'min_q2_value': np.min(q2_values),
            'target_q_avg': np.mean(target_q_values),
            'entropy': entropy,
            'batch_reward_avg': np.mean(rewards),
            'batch_reward_min': np.min(rewards),
            'batch_reward_max': np.max(rewards),
            'batch_size': len(minibatch),
            'mpc_critic_epsilon': self.mpc_critic_epsilon
        }

        if self.trainer_iteration % settings.TD3_DELAYED_POLICY_UPDATE == 0:
            logs.update({'actor_loss': actor_loss.numpy(), 
                         'actor_gradient_mean': np.mean(actor_grad_norms),
                          'actor_gradient_max': np.max(actor_grad_norms), 
                          'actor_gradient_min': np.min(actor_grad_norms)})
        
        if settings.MPC_PHASE == 'Transition':
            self.mpc_critic_epsilon = max(settings.MPC_CRITIC_EPSILON_MIN, self.mpc_critic_epsilon * settings.MPC_CRITIC_EPSILON_DECAY)

        if indices is not None:
            self.experience_replay.update_priorities(indices, td_errors.numpy())

        if self.optimizer[2] == 1:
            self.optimizer[2] = 0
            #self.compile_model(model=self.critic_model, learning_rate=self.optimizer[3], decay=self.get_lr_decay('critic')[1])
            self.critic_optimizer.learning_rate.assign(self.optimizer[3])
        if self.optimizer[4] == 1:
            self.optimizer[4] = 0
            # implement decay if needed
            #self.compile_model(model=self.critic_model, learning_rate=self.get_lr_decay('critic')[0], decay=self.optimizer[5])
            #self.optimizer[0], self.optimizer[1] = self.get_lr_decay('critic')

        if self.optimizer[8] == 1:
            self.optimizer[8] = 0
            self.actor_optimizer.learning_rate.assign(self.optimizer[9])
        if self.optimizer[10] == 1:
            self.optimizer[10] = 0
            # implement decay if needed

        return logs
    

    def train(self):
        """Train method called in a loop. Extracts and prepares the transitions from the experience replay and calls the chosen RL-Training method."""
        if self.experience_replay.get_memory_size() < settings.MIN_EXPERIENCE_REPLAY_SIZE:
            return False # Ensure enough experiences before training starts
        
        self.trainer_iteration += 1
        # old experiece replay method doesn't update is_weights
        if settings.EXPERIENCE_REPLAY_METHOD == 'reward_old':
            minibatch = self.experience_replay.sample(settings.MINIBATCH_SIZE)
            indices, is_weights = None, None
        else:
            try:
                minibatch, indices, is_weights = self.experience_replay.sample(settings.MINIBATCH_SIZE)
                indices = np.array(indices)
                is_weights = np.array(is_weights, dtype=np.float32).reshape(-1, 1)
            except:
                return True

        # extract states from minibatch and stack them for training
        current_states = [transition[0][0] for transition in minibatch]  # State before taking action
        next_states = [transition[0][1] for transition in minibatch]  # State after taking action
        actions = np.array([transition[1] for transition in minibatch])  # Actions taken
        rewards = np.array([transition[2] for transition in minibatch])  # Rewards received
        dones = np.array([transition[3] for transition in minibatch])    # Done flags

        mpc_actions = None
        if settings.USE_MPC:
            mpc_actions = np.array([transition[4] for transition in minibatch])

        current_states_stacked = [np.vstack([x[i] for x in current_states]) for i in range(len(current_states[0]))]
        next_states_stacked = [np.vstack([x[i] for x in next_states]) for i in range(len(next_states[0]))]


        # DQN
        if self.rl_algorithm == 'dqn':
            logs = self.train_dqn(current_states_stacked, next_states_stacked, actions, rewards, dones, minibatch)
        # DDPG
        elif self.rl_algorithm == 'ddpg':
            logs = self.train_ddpg(current_states_stacked, next_states_stacked, actions, mpc_actions, rewards, dones, minibatch, indices, is_weights)
        # TD3
        elif self.rl_algorithm == 'td3':
            logs = self.train_td3(current_states_stacked, next_states_stacked, actions, mpc_actions, rewards, dones, minibatch, indices, is_weights)
        
        # log trainer stats
        self.tensorboard.update_stats('trainer', **logs)
        if self.neptune_logger:
            self.neptune_logger.update_stats('trainer', **logs)

        return True

    
    
    def train_in_loop(self):
        """calls training and is responsible for model saving."""
        self.tps_counter = deque(maxlen=20)

        while True:
            
            step_start = time.time()

            if self.stop.value == STOP.stopping:
                return

            if self.stop.value in [STOP.carla_simulator_error, STOP.restarting_carla_simulator]:
                self.trainer_stats[0] = TRAINER_STATE.paused
                time.sleep(1)
                continue

           
            if not self.train():
                self.trainer_stats[0] = TRAINER_STATE.waiting

                if self.stop.value in [STOP.at_checkpoint, STOP.now]:
                    self.stop.value = STOP.stopping

                time.sleep(0.01)
                continue

            self.trainer_stats[0] = TRAINER_STATE.training

            # Share new weights with models as fast as possible
            self.weights.raw = self.serialize_weights()

            with self.weights_iteration.get_lock():
                self.weights_iteration.value += 1

            frame_time = time.time() - step_start
            self.tps_counter.append(frame_time)
            self.trainer_stats[1] = len(self.tps_counter)/sum(self.tps_counter)

            # if model performed good
            save_model = self.save_model
            if save_model:
                if self.rl_algorithm == 'dqn':
                    self.model.save(save_model)
                elif self.rl_algorithm == 'ddpg':
                    self.actor_model.save(save_model + '_actor')
                    self.critic_model.save(save_model + '_critic')
                elif self.rl_algorithm == 'td3':
                    self.actor_model.save(save_model + '_actor')
                    self.critic_model1.save(save_model + '_critic1')
                    self.critic_model2.save(save_model + '_critic2')
                self.save_model = False


            checkpoint_number = self.episode.value // self.save_checkpoint_every.value

            # save hparams and model
            if checkpoint_number > self.last_checkpoint or self.stop.value == STOP.now:
                self.models.append(f'checkpoint/{settings.MODEL_NAME}_{self.episode.value}')
                hparams = {
                    'duration': self.duration.value,
                    'episode': self.episode.value,
                    'trainer_iteration': self.trainer_iteration,
                    'epsilon': list(self.epsilon),
                    'discount': self.discount.value,
                    'rl_algorithm': self.rl_algorithm,
                    'update_target_every': self.update_target_every.value,
                    'last_target_update': self.last_target_update,
                    'min_reward': self.min_reward.value,
                    'agent_show_preview': [list(preview) for preview in self.agent_show_preview],
                    'save_checkpoint_every': self.save_checkpoint_every.value,
                    'sync_mode' : self.sync_mode,
                    'use_n_future_states' : self.use_n_future_steps,
                    'put_trans_every' : self.put_trans_every,
                    'model_path': f'checkpoint/{settings.MODEL_NAME}_{self.episode.value}',
                    'logdir': self.logdir,
                    'weights_iteration': self.weights_iteration.value,
                    'car_npcs': list(self.car_npcs),
                    'models': list(set(self.models))
                }

                if self.sync_mode:
                    hparams['fixed_delta_sec'] =  self.fixed_delta_seconds
                    hparams['steps_per_episode'] = self.steps_per_episode.value
                else:
                    hparams['seconds_per_episode'] = self.seconds_per_episode.value

                hparams['mpc_critic_epsilon'] = self.mpc_critic_epsilon
                hparams['mpc_critic_epsilon_decay'] = settings.MPC_CRITIC_EPSILON_DECAY

                if self.rl_algorithm == 'dqn':
                    self.model.save(f'checkpoint/{settings.MODEL_NAME}_{hparams["episode"]}_dqn.model')
                elif self.rl_algorithm == 'ddpg':
                    self.actor_model.save(f'checkpoint/{settings.MODEL_NAME}_{hparams["episode"]}_ddpg_actor.model')
                    self.critic_model.save(f'checkpoint/{settings.MODEL_NAME}_{hparams["episode"]}_ddpg_critic.model')
                elif self.rl_algorithm == 'td3':
                    self.actor_model.save(f'checkpoint/{settings.MODEL_NAME}_{hparams["episode"]}_td3_actor.model')
                    self.critic_model1.save(f'checkpoint/{settings.MODEL_NAME}_{hparams["episode"]}_td3_critic1.model')
                    self.critic_model2.save(f'checkpoint/{settings.MODEL_NAME}_{hparams["episode"]}_td3_critic2.model')
                
                

                with open('checkpoint/hparams_new.json', 'w', encoding='utf-8') as f:
                    json.dump(hparams, f)
                try:
                    os.remove('checkpoint/hparams.json')
                except:
                    pass
                try:
                    os.rename('checkpoint/hparams_new.json', 'checkpoint/hparams.json')
                    self.last_checkpoint = checkpoint_number
                except Exception as e:
                    print(str(e))

                
                if (self.stop.value == STOP.now or self.stop.value == STOP.at_checkpoint) and settings.ADDITIONAL_SAVE_FOLDER_PATH:
                    dst_dir = settings.ADDITIONAL_SAVE_FOLDER_PATH
                    i = 1
                    while os.path.isdir(dst_dir):
                        dst_dir = settings.ADDITIONAL_SAVE_FOLDER_PATH + f"({i})"
                        i += 1
                    
                    src_dir = 'checkpoint'
                    shutil.copytree(src_dir, dst_dir)
                    
                    src_dir = 'models'
                    models = os.listdir(src_dir)

                    cur_timestamp = 0
                    cur_model = None
                    for model in models:
                        timestamp = int(model.split('_')[-1].replace('timestamp.model',''))
                        if timestamp > cur_timestamp:
                            cur_timestamp = timestamp
                            cur_model = model 
                        
                    if cur_model:
                        shutil.copytree(os.path.join(src_dir, cur_model), dst_dir)  
                        
                    src_file = self.logdir
                    
                    shutil.copy(src_file, dst_dir)
                    
            if self.stop.value in [STOP.at_checkpoint, STOP.now]:
                self.stop.value = STOP.stopping





def log_init_parameters(trainer):
    """logs initial Parameters for the environment, sensors, setup"""
    hyperparameters = {
    'continued_training': settings.USE_HPARAMS,  
    'reset_car_npc_every_n_ticks': settings.RESET_CAR_NPC_EVERY_N_TICKS,
    'rotate_map_every': settings.ROTATE_MAP_EVERY,
    'dynamic_weather': settings.DYNAMIC_WEATHER,
    'weather_preset': settings.WEATHER_PRESET,
    'disallowed_npc_vehicles': settings.DISALLOWED_NPC_VEHICLES,
    'rl_algorithm': trainer.rl_algorithm,
    'actions': settings.ACTIONS if trainer.rl_algorithm == 'dqn' else "continous",
    'collision_filter': settings.COLLISION_FILTER,
    'weight_rewards_with_speed': settings.WEIGHT_REWARDS_WITH_SPEED,
    'max_speed': settings.MAX_SPEED,
    'speed_max_reward': settings.SPEED_MAX_REWARD,
    'speed_min_reward': settings.SPEED_MAX_PENALTY,
    'collision_penalty': settings.COLLISION_PENALTY,
    'agents': settings.AGENTS,
    'vehicle': settings.VEHICLE,
    'update_weights_every': settings.UPDATE_WEIGHTS_EVERY,
    'minibatch_size': settings.MINIBATCH_SIZE,
    'training_batch_size': settings.TRAINING_BATCH_SIZE,
    'prediction_batch_size': settings.PREDICTION_BATCH_SIZE,
    'update_target_every': settings.UPDATE_TARGET_EVERY,
    'aggregate_stats_every': settings.AGGREGATE_STATS_EVERY,
    'discount': settings.DISCOUNT,
    'use_n_future_steps': settings.USE_N_FUTURE_STEPS,
    'put_trans_every': settings.PUT_TRANS_EVERY,
    'sync_mode' : trainer.sync_mode,
    'put_trans_every' : trainer.put_trans_every,
    'model_name': settings.MODEL_NAME,
    'use_mpc': settings.USE_MPC,
    'discount': trainer.discount.value,
    'optimizer': 'Adam',
    'loss': 'MSE',
    'max_distance_for_lane_center_reward': settings.MAX_DISTANCE_FOR_LANE_CENTER_REWARD,
    'distance_for_max_lane_center_penalty': settings.DISTANCE_FOR_MAX_LANE_CENTER_PENALTY,
    'lane_center_max_reward': settings.LANE_CENTER_MAX_REWARD,
    'lane_center_max_penalty': settings.LANE_CENTER_MAX_PENALTY,
    'max_distance_before_route_left': settings.MAX_DISTANCE_BEFORE_ROUTE_LEFT,
    'route_left_penalty': settings.ROUTE_LEFT_PENALTY,
    'yaw_error_max_reward': settings.YAW_ERROR_MAX_REWARD,
    'yaw_error_max_penalty': settings.YAW_ERROR_MAX_PENALTY,
    'max_yaw_error_threshhold': settings.MAX_YAW_ERROR_THRESHOLD,
    'max_yaw_penalty_error': settings.YAW_PENALTY_ERROR_MAX,
    'waypoint_reached_reward': settings.WAYPOINT_REACHED_REWARD,
    'waypoint_missed_reward': settings.WAYPOINT_MISSED_PENALTY

}
    
    if trainer.rl_algorithm == 'dqn':
        hyperparameters['start_epsilon'] = settings.START_EPSILON
        hyperparameters['epsilon_decay'] = settings.EPSILON_DECAY
        hyperparameters['min_epsilon'] = settings.MIN_EPSILON
        hyperparameters['smoothened_actions'] = settings.SMOOTH_ACTIONS
        hyperparameters['optimizer_learning_rate'] = settings.DQN_OPTIMIZER_LEARNING_RATE
        hyperparameters['optimizer_decay'] = settings.DQN_OPTIMIZER_DECAY
    else:
        hyperparameters['noise_type'] = settings.NOISE_TYPE
        hyperparameters['preprocess_action_input'] = settings.DDPG_PREPROCESS_ACTION_INPUT
        hyperparameters['critic_learning_rate'] = settings.DDPG_CRITIC_OPTIMIZER_LEARNING_RATE
        hyperparameters['actor_learning_rate'] = settings.DDPG_ACTOR_OPTIMIZER_LEARNING_RATE
        hyperparameters['critic_optimizer_decay'] = settings.DDPG_CRITIC_OPTIMIZER_DECAY
        hyperparameters['actor_optimizer_decay'] = settings.DDPG_ACTOR_OPTIMIZER_DECAY
        
        for k, v in settings.NOISE_THROTTLE_PARAMS.items():
            hyperparameters[k] = v

        for k, v in settings.NOISE_STEERING_PARAMS.items():
            hyperparameters[k] = v


    if trainer.sync_mode:
        hyperparameters['fixed_delta_sec'] =  trainer.fixed_delta_seconds
        hyperparameters['steps_per_episode'] = trainer.steps_per_episode.value
    else:
        hyperparameters['seconds_per_episode'] = trainer.seconds_per_episode.value
    
    trainer.tensorboard.log_init("hyperparameters",**hyperparameters)
    if 'neptune' in settings.ADDITIONAL_LOGGING:
        trainer.neptune_logger.log_init("hyperparameters",**hyperparameters)
    
    
    sensor_settings = {
        
        'main_cam': settings.FRONT_CAM_TYPE,
        'collision_detector': True,  
              

    }

    for k, v in settings.REWARD_FUNCTION_METRICS.items():
        if v:
            sensor_settings.update({f'reward_function_metrics/{k}': v})

    for k, v in settings.MODEL_INPUTS.items():
        if v:
            sensor_settings.update({f'model_inputs/{k}': v})    

    if settings.REWARD_FUNCTION_METRICS['move']:
        sensor_settings.update({'move_penalty': settings.MOVE_PENALTY})
    if settings.FRONT_CAM_TYPE == 'rgb' and settings.MODEL_INPUTS['front_camera']:
        sensor_settings.update({
            'rgb/img_width': settings.RGB_CAM_SETTINGS[0],
            'rgb/img_height': settings.RGB_CAM_SETTINGS[1],
            'rgb/pos_x': settings.RGB_CAM_SETTINGS[2],
            'rgb/pos_y': settings.RGB_CAM_SETTINGS[3],
            'rgb/pos_z': settings.RGB_CAM_SETTINGS[4],
            'rgb/fov': settings.RGB_CAM_SETTINGS[5],
            'rgb/img_type': settings.RGB_CAM_SETTINGS[6],
            'rgb/bloom_intensity': settings.RGB_CAM_SETTINGS[7],
            'rgb/fstop': settings.RGB_CAM_SETTINGS[8],
            'rgb/iso': settings.RGB_CAM_SETTINGS[9],
            'rgb/gamma': settings.RGB_CAM_SETTINGS[10],
            'rgb/lens_flare_intensity': settings.RGB_CAM_SETTINGS[11],
            'rgb/sensor_tick': settings.RGB_CAM_SETTINGS[12],
            'rgb/shutter_speed': settings.RGB_CAM_SETTINGS[13],
            'rgb/lens_circle_falloff': settings.RGB_CAM_SETTINGS[14],
            'rgb/lens_circle_multiplier': settings.RGB_CAM_SETTINGS[15],
            'rgb/lens_k': settings.RGB_CAM_SETTINGS[16],
            'rgb/lens_kcube': settings.RGB_CAM_SETTINGS[17],
            'rgb/lens_x_size': settings.RGB_CAM_SETTINGS[18],
            'rgb/lens_y_size': settings.RGB_CAM_SETTINGS[19],
        })

    if settings.FRONT_CAM_TYPE == 'depth' and settings.MODEL_INPUTS['front_camera']:
        sensor_settings.update({
            'depth/img_width': settings.DEPTH_CAM_SETTINGS[0],
            'depth/img_height': settings.DEPTH_CAM_SETTINGS[1],
            'depth/pos_x': settings.DEPTH_CAM_SETTINGS[2],
            'depth/pos_y': settings.DEPTH_CAM_SETTINGS[3],
            'depth/pos_z': settings.DEPTH_CAM_SETTINGS[4],
            'depth/fov': settings.DEPTH_CAM_SETTINGS[5],
            'depth/sensor_tick': settings.DEPTH_CAM_SETTINGS[6],
            'depth/lens_circle_falloff': settings.DEPTH_CAM_SETTINGS[7],
            'depth/lens_circle_multiplier': settings.DEPTH_CAM_SETTINGS[8],
            'depth/lens_k': settings.DEPTH_CAM_SETTINGS[9],
            'depth/lens_kcube': settings.DEPTH_CAM_SETTINGS[10],
            'depth/lens_x_size': settings.DEPTH_CAM_SETTINGS[11],
            'depth/lens_y_size': settings.DEPTH_CAM_SETTINGS[12],
        })

    if settings.FRONT_CAM_TYPE == 'semseg' and settings.MODEL_INPUTS['front_camera']:
        sensor_settings.update({
            'semantic/img_width': settings.SEMANTIC_CAM_SETTINGS[0],
            'semantic/img_height': settings.SEMANTIC_CAM_SETTINGS[1],
            'semantic/pos_x': settings.SEMANTIC_CAM_SETTINGS[2],
            'semantic/pos_y': settings.SEMANTIC_CAM_SETTINGS[3],
            'semantic/pos_z': settings.SEMANTIC_CAM_SETTINGS[4],
            'semantic/fov': settings.SEMANTIC_CAM_SETTINGS[5],
            'semantic/sensor_tick': settings.SEMANTIC_CAM_SETTINGS[6],
            'semantic/lens_circle_falloff': settings.SEMANTIC_CAM_SETTINGS[7],
            'semantic/lens_circle_multiplier': settings.SEMANTIC_CAM_SETTINGS[8],
            'semantic/lens_k': settings.SEMANTIC_CAM_SETTINGS[9],
            'semantic/lens_kcube': settings.SEMANTIC_CAM_SETTINGS[10],
            'semantic/lens_x_size': settings.SEMANTIC_CAM_SETTINGS[11],
            'semantic/lens_y_size': settings.SEMANTIC_CAM_SETTINGS[12],
        })

    if settings.FRONT_CAM_TYPE == 'simseg' and settings.MODEL_INPUTS['front_camera']:
        sensor_settings.update({
            'instance/img_width': settings.INSTANCE_CAM_SETTINGS[0],
            'instance/img_height': settings.INSTANCE_CAM_SETTINGS[1],
            'instance/pos_x': settings.INSTANCE_CAM_SETTINGS[2],
            'instance/pos_y': settings.INSTANCE_CAM_SETTINGS[3],
            'instance/pos_z': settings.INSTANCE_CAM_SETTINGS[4],
            'instance/fov': settings.INSTANCE_CAM_SETTINGS[5],
            'instance/sensor_tick': settings.INSTANCE_CAM_SETTINGS[6],
            'instance/lens_circle_falloff': settings.INSTANCE_CAM_SETTINGS[7],
            'instance/lens_circle_multiplier': settings.INSTANCE_CAM_SETTINGS[8],
            'instance/lens_k': settings.INSTANCE_CAM_SETTINGS[9],
            'instance/lens_kcube': settings.INSTANCE_CAM_SETTINGS[10],
            'instance/lens_x_size': settings.INSTANCE_CAM_SETTINGS[11],
            'instance/lens_y_size': settings.INSTANCE_CAM_SETTINGS[12],
        })

    if settings.REWARD_FUNCTION_METRICS['imu']:
        sensor_settings.update({
            'imu/pos_x': settings.IMU_SETTINGS[0],
            'imu/pos_y': settings.IMU_SETTINGS[1],
            'imu/pos_z': settings.IMU_SETTINGS[2],
            'imu/sensor_tick': settings.IMU_SETTINGS[3],
            'imu/noise_accel_stddev_x': settings.IMU_SETTINGS[4],
            'imu/noise_accel_stddev_y': settings.IMU_SETTINGS[5],
            'imu/noise_accel_stddev_z': settings.IMU_SETTINGS[6],
            'imu/noise_gyro_bias_x': settings.IMU_SETTINGS[7],
            'imu/noise_gyro_bias_y': settings.IMU_SETTINGS[8],
            'imu/noise_gyro_bias_z': settings.IMU_SETTINGS[9],
            'imu/noise_gyro_stddev_x': settings.IMU_SETTINGS[10],
            'imu/noise_gyro_stddev_y': settings.IMU_SETTINGS[11],
            'imu/noise_gyro_stddev_z': settings.IMU_SETTINGS[12],
            'imu/noise_seed': settings.IMU_SETTINGS[13],
        })

    if settings.REWARD_FUNCTION_METRICS['lane_invasion'] or settings.MODEL_INPUTS['lane_invasion']:
        sensor_settings.update({
            'lane/pos_x': settings.LANE_INVASION_SETTINGS[0],
            'lane/pos_y': settings.LANE_INVASION_SETTINGS[1],
            'lane/pos_z': settings.LANE_INVASION_SETTINGS[2],
        })

        for k, v in settings.LANE_INVASION_FILTER:
            sensor_settings.update({f'lane/{k}': v})

    if settings.MODEL_INPUTS['radar']:
        sensor_settings.update({
            'radar/pos_x': settings.RADAR_SETTINGS[0],
            'radar/pos_y': settings.RADAR_SETTINGS[1],
            'radar/pos_z': settings.RADAR_SETTINGS[2],
            'radar/fov_hor': settings.RADAR_SETTINGS[3],
            'radar/fov_ver': settings.RADAR_SETTINGS[4],
            'radar/range': settings.RADAR_SETTINGS[5],
            'radar/pps': settings.RADAR_SETTINGS[6],
            'radar/sensor_tick': settings.RADAR_SETTINGS[7],
        })

    if settings.MODEL_INPUTS['lidar']:
        sensor_settings.update({
            'lidar/pos_x': settings.LIDAR_SETTINGS[0],
            'lidar/channels': settings.LIDAR_SETTINGS[1],
            'lidar/range': settings.LIDAR_SETTINGS[2],
            'lidar/points_per_second': settings.LIDAR_SETTINGS[3],
            'lidar/rotation_frequency': settings.LIDAR_SETTINGS[4],
            'lidar/upper_fov': settings.LIDAR_SETTINGS[5],
            'lidar/lower_fov': settings.LIDAR_SETTINGS[6],
            'lidar/horizontal_fov': settings.LIDAR_SETTINGS[7],
            'lidar/atmosphere_attenuation_rate': settings.LIDAR_SETTINGS[8],
            'lidar/dropoff_general_rate': settings.LIDAR_SETTINGS[9],
            'lidar/dropoff_intensity_limit': settings.LIDAR_SETTINGS[10],
            'lidar/dropoff_zero_intensity': settings.LIDAR_SETTINGS[11],
            'lidar/sensor_tick': settings.LIDAR_SETTINGS[12],
        })

    if settings.MODEL_INPUTS['obstacle']:
        sensor_settings.update({
            'obstacle/pos_x': settings.OBSTACLE_DETECTOR_SETTINGS[0],
            'obstacle/pos_y': settings.OBSTACLE_DETECTOR_SETTINGS[1],
            'obstacle/pos_z': settings.OBSTACLE_DETECTOR_SETTINGS[2],
            'obstacle/distance': settings.OBSTACLE_DETECTOR_SETTINGS[3],
            'obstacle/hit_radius': settings.OBSTACLE_DETECTOR_SETTINGS[4],
            'obstacle/only_dynamics': settings.OBSTACLE_DETECTOR_SETTINGS[5],
            'obstacle/sensor_tick': settings.OBSTACLE_DETECTOR_SETTINGS[6],
        })

    if settings.REWARD_FUNCTION_METRICS['waypoint_reached'] or settings.MODEL_INPUTS['relative_pos']:
        sensor_settings.update({
            'gnss/noise_alt_bias': settings.GNSS_SENSOR_SETTINGS[0],
            'gnss/noise_alt_stddev': settings.GNSS_SENSOR_SETTINGS[1],
            'gnss/noise_lat_bias': settings.GNSS_SENSOR_SETTINGS[2],
            'gnss/noise_lat_stddev': settings.GNSS_SENSOR_SETTINGS[3],
            'gnss/noise_lon_bias': settings.GNSS_SENSOR_SETTINGS[4],
            'gnss/noise_lon_stddev': settings.GNSS_SENSOR_SETTINGS[5],
            'gnss/noise_seed': settings.GNSS_SENSOR_SETTINGS[6],
            'gnss/sensor_tick': settings.GNSS_SENSOR_SETTINGS[7],
            'navigation/distance_to_goal': settings.DISTANCE_TO_GOAL,
            'navigation/distance_between_waypoints': settings.DISTANCE_BETWEEN_WAYPOINTS,
            'navigation/waypoint_radius': settings.WAYPOINT_RADIUS,
            'navigation/waypoint_reached_reward': settings.WAYPOINT_REACHED_REWARD,
            'navigation/goal_reached_reward': settings.GOAL_REACHED_REWARD        
                    })


    trainer.tensorboard.log_init("sensors",**sensor_settings)
    if 'neptune' in settings.ADDITIONAL_LOGGING:
        trainer.neptune_logger.log_init("sensors",**sensor_settings)

@dataclass(frozen=True)
class TRAINER_STATE:
    starting:int = 0
    waiting:int = 1
    training:int = 2
    finished:int = 3
    paused:int = 4


TRAINER_STATE_MESSAGE = {
    TRAINER_STATE.starting: 'STARTING',
    TRAINER_STATE.waiting: 'WAITING',
    TRAINER_STATE.training: 'TRAINING',
    TRAINER_STATE.finished: 'FINISHED',
    TRAINER_STATE.paused: 'PAUSED',
}


def run(model_path, logdir, stop, weights, weights_iteration, episode, rl_algorithm, trainer_iteration, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, steps_per_episode, duration, transitions, tensorboard_stats, trainer_stats, episode_stats, optimizer, models, car_npcs, carla_settings_stats, carla_fps, sync_mode, fixed_delta_seconds, use_n_future_states, put_trans_every):    
    """Creates and runs trainer process. Logs Agent and Simulation stats and can call for model save."""
    
    # Set GPU for TF
    if not settings.USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif settings.TRAINER_GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= str(settings.TRAINER_GPU)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                memory_limit = int(settings.TRAINER_MEMORY) 
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
        except Exception as e:
              print(e)
    
    tf.random.set_seed(42)
    random.seed(42)
    np.random.seed(42)  


    

    # Create and initialize Trainer
    trainer = ARTDQNTrainer(model_path, rl_algorithm)
    trainer.init_training(stop, logdir, trainer_stats, episode, trainer_iteration, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, steps_per_episode, duration, optimizer, models, car_npcs, sync_mode, fixed_delta_seconds, use_n_future_states, put_trans_every)    
    trainer.init_serialized_weights(weights, weights_iteration)
    trainer_stats[0] = TRAINER_STATE.waiting
    
    log_init_parameters(trainer)

    trainer_thread = Thread(target=trainer.train_in_loop, daemon=True)
    trainer_thread.start()

    aggregate_stats_every = settings.AGGREGATE_STATS_EVERY
    raw_rewards = deque(maxlen=settings.AGENTS*aggregate_stats_every)
    weighted_rewards = deque(maxlen=settings.AGENTS*aggregate_stats_every)
    episode_times = deque(maxlen=settings.AGENTS*aggregate_stats_every)
    frame_times = deque(maxlen=settings.AGENTS)
    episode_end_reasons = deque(maxlen=settings.AGENTS*aggregate_stats_every)

    configured_actions = [getattr(ACTIONS, action) for action in settings.ACTIONS]
    min_reward = settings.MIN_REWARD
    min_avg_reward = settings.MIN_AVG_REWARD
    
    
    

    while stop.value != 3:
        
        if episode.value > trainer.tensorboard.step:
            trainer.tensorboard.step = episode.value
            if 'neptune' in settings.ADDITIONAL_LOGGING:
                trainer.neptune_logger.step = episode.value
        
        for _ in range(transitions.qsize()):
            try:
                trainer.update_experience_replay(transitions.get(True, 0.1))
            except:
                break
        

        # logging the info the agents provide
        while not tensorboard_stats.empty():
            # Added to a Queue by agents
            if trainer.rl_algorithm == 'dqn':
                agent_episode, reward, per_step_avg_reward, agent_epsilon, episode_time, frame_time, weighted_reward, predicted_actions, random_actions, avg_pred_time, avg_speed, waypoints_reached, waypoints_skipped, waypoints_total, waypoints_total_per, episode_end_reason, step, per_step_rewards, step_time, action_step_time, mpc_step_time, sp, avg_lateral_error, avg_yaw_error, mpc_exploration_epsilon, mpc_action_counter, *avg_predicted_qs = tensorboard_stats.get_nowait()
            else:
                agent_episode, reward, per_step_avg_reward, episode_time, frame_time, weighted_reward, avg_pred_time, avg_speed, waypoints_reached, waypoints_skipped, waypoints_total, waypoints_total_per, episode_end_reason, step, per_step_rewards, step_time, action_step_time, mpc_step_time, sp, avg_lateral_error, avg_yaw_error, mpc_exploration_epsilon, mpc_action_counter, episode_throttle_values, episode_steering_values, episode_throttle_noise, episode_steering_noise, episode_applied_throttle_values, episode_applied_steering_values, action_magnitudes, action_deltas, noise_magnitudes, current_throttle_sigma, current_steering_sigma = tensorboard_stats.get_nowait()

            raw_rewards.append(reward)
            weighted_rewards.append(weighted_reward)
            episode_times.append(episode_time)
            frame_times.append(frame_time)
            episode_end_reasons.append(episode_end_reason)

            episode_stats[0] = min(raw_rewards)  # Minimum reward (raw)
            episode_stats[1] = sum(raw_rewards)/len(raw_rewards)  # Average reward (raw)
            episode_stats[2] = max(raw_rewards)  # Maximum reward (raw)
            episode_stats[3] = min(episode_times)  # Minimum episode duration
            episode_stats[4] = sum(episode_times)/len(episode_times)  # Average episode duration
            episode_stats[5] = max(episode_times)  # Maximum episode duration
            episode_stats[6] = sum(frame_times)/len(frame_times)  # Average agent FPS
            episode_stats[7] = min(weighted_rewards)  # Minimum reward (weighted)
            episode_stats[8] = sum(weighted_rewards)/len(weighted_rewards)  # Average reward (weighted)
            episode_stats[9] = max(weighted_rewards)  # Maximum reward (weighted)
            experience_replay_size = trainer.experience_replay.get_memory_size()
            

            reward_std = np.std(raw_rewards)
            weighted_reward_std = np.std(weighted_rewards)
            
            episode_end_counts = [0, 0, 0, 0] 

           
            for i in episode_end_reasons:
                if 0 <= i <= 3:  
                    episode_end_counts[i] += 1


            carla_stats = {}
            for process_no in range(settings.CARLA_HOSTS_NO):
                for index, stat in enumerate(['carla_{}_car_npcs', 'carla_{}_weather_sun_azimuth', 'carla_{}_weather_sun_altitude', 'carla_{}_weather_clouds_pct', 'carla_{}_weather_wind_pct', 'carla_{}_weather_rain_pct']):
                    if carla_settings_stats[process_no][index] != -1:
                        carla_stats[stat.format(process_no+1)] = carla_settings_stats[process_no][index]
                carla_stats[f'carla_{process_no + 1}_fps'] = carla_fps[process_no].value
            
            stats = ({
            'step': agent_episode,
            'episode': agent_episode,
            'episode_total_reward': reward,
            'per_step_total_rewards': reward / step,
            'reward_raw_avg': episode_stats[1],
            'reward_raw_min': episode_stats[0],
            'reward_raw_max': episode_stats[2],
            'reward_raw_std': reward_std,
            'reward_weighted_avg': episode_stats[8] if not sync_mode else -1,
            'reward_weighted_min': episode_stats[7] if not sync_mode else -1,
            'reward_weighted_max': episode_stats[9] if not sync_mode else -1,
            'reward_weighted_std': weighted_reward_std if not sync_mode else -1,
            'avg_reward': per_step_avg_reward,
            'episode_time_avg': episode_stats[4],
            'episode_time_min': episode_stats[3],
            'episode_time_max': episode_stats[5],
            'time_per_step': step_time,
            'time_per_action': action_step_time,
            'time_per_mpc': mpc_step_time,
            'agent_fps_avg': episode_stats[6],
            'optimizer_lr': optimizer[0],
            'optimizer_decay': optimizer[1],
            'experience_replay_size': experience_replay_size,
            'avg_pred_time' : avg_pred_time,
            'actions_taken': step,
            'avg_speed': avg_speed,
            'waypoints_reached': waypoints_reached,
            'waypoints_skipped': waypoints_skipped,
            'waypoints_total': waypoints_total,
            'waypoints_total_per': waypoints_total_per,
            'end_collision': episode_end_counts[0] / len(episode_end_reasons),
            'end_reached_goal': episode_end_counts[1] / len(episode_end_reasons),
            'end_left_route': episode_end_counts[2] / len(episode_end_reasons),
            'end_criterium_reached': episode_end_counts[3] / len(episode_end_reasons),
            'spawn_point': sp,
            'avg_lateral_error': avg_lateral_error,
            'avg_yaw_error': avg_yaw_error,
            'mpc_exploration_epsilon': mpc_exploration_epsilon,
            'mpc_actions_per': mpc_action_counter / step,
            **carla_stats
        })

            for k, v in per_step_rewards.items():
                stats[k] = v / step

            if trainer.rl_algorithm == 'dqn':

                tensorboard_q_stats = {}
                for action, (avg_predicted_q, std_predicted_q, usage_predicted_q) in enumerate(zip(avg_predicted_qs[0::3], avg_predicted_qs[1::3], avg_predicted_qs[2::3])):
                    if avg_predicted_q != -10**6:
                        episode_stats[10 + action*3] = avg_predicted_q
                        tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_avg' if action else f'q_all_actions_avg'] = avg_predicted_q
                    if std_predicted_q != -10 ** 6:
                        episode_stats[11 + action*3] = std_predicted_q
                        tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_std' if action else f'q_all_actions_std'] = std_predicted_q
                    if usage_predicted_q != -10 ** 6:
                        episode_stats[12 + action*3] = usage_predicted_q
                        if action > 0:
                            tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_usage_pct'] = usage_predicted_q
               


                predicted_actions_pct = predicted_actions / max((predicted_actions + random_actions), 1)
            
                stats.update({
                    'epsilon': agent_epsilon,
                    'predicted_actions_pct': predicted_actions_pct,
                    **tensorboard_q_stats,
                })
        

            else:

                stats.update({
                    'min_action_magnitude': 0 if len(action_magnitudes) < 1 else np.min(action_magnitudes),
                    'max_action_magnitude': 0 if len(action_magnitudes) < 1 else np.max(action_magnitudes),
                    'mean_action_magnitude': 0 if len(action_magnitudes) < 1 else np.mean(action_magnitudes),
                    'std_action_magnitude': 0 if len(action_magnitudes) < 1 else np.std(action_magnitudes),

                    'min_action_delta': 0 if len(action_deltas) < 1 else np.min(action_deltas),
                    'max_action_delta': 0 if len(action_deltas) < 1 else  np.max(action_deltas),
                    'mean_action_delta': 0 if len(action_deltas) < 1 else np.mean(action_deltas),
                    'std_action_delta': 0 if len(action_deltas) < 1 else np.std(action_deltas),

                    'min_noise_magnitude': 0 if len(noise_magnitudes) < 1 else np.min(noise_magnitudes),
                    'max_noise_magnitude': 0 if len(noise_magnitudes) < 1 else np.max(noise_magnitudes),
                    'mean_noise_magnitude': 0 if len(noise_magnitudes) < 1 else np.mean(noise_magnitudes),
                    'std_noise_magnitude': 0 if len(noise_magnitudes) < 1 else np.std(noise_magnitudes),

                    'throttle_sigma': current_throttle_sigma,
                    'steering_sigma': current_steering_sigma

})
                if 'neptune' in settings.ADDITIONAL_LOGGING and (agent_episode < 50 or agent_episode % 10 == 0):
                    inputs_predicted = pd.DataFrame({
                        "timestep": list(range(len(episode_throttle_values))),
                        "throttle": episode_throttle_values,
                        "steering": episode_steering_values,
                    })
                    trainer.neptune_logger.log_df('agent/inputs_predicted', f'inputs_predicted_{agent_episode}', inputs_predicted)

                    inputs_noise = pd.DataFrame({
                        "timestep": list(range(len(episode_throttle_noise))),
                        "throttle": episode_throttle_noise,
                        "steering": episode_steering_noise,
                    })
                    trainer.neptune_logger.log_df('agent/inputs_noise', f'inputs_noise_{agent_episode}', inputs_noise)

                    inputs_applied = pd.DataFrame({
                        "timestep": list(range(len(episode_applied_throttle_values))),
                        "throttle": episode_applied_throttle_values,
                        "steering": episode_applied_steering_values,
                    })
                    trainer.neptune_logger.log_df('agent/inputs_applied', f'inputs_applied_{agent_episode}', inputs_applied)

             
            trainer.tensorboard.update_stats("agent",**stats)
            if 'neptune' in settings.ADDITIONAL_LOGGING:
                trainer.neptune_logger.update_stats("agent",**stats)
            

            # if model performs good, call for model save
            if sync_mode and (episode_stats[0] >= min_reward or episode_stats[1] >= min_avg_reward):
                trainer.save_model = f'models/{settings.MODEL_NAME}_sync{sync_mode}_{episode_stats[1]:_>7.2f}avg_{episode_stats[2]:_>7.2f}max_{episode_stats[0]:_>7.2f}min_{agent_episode}episode_{int(time.time())}timestamp.model'
                min_reward = episode_stats[0]
                min_avg_reward = episode_stats[1]
            elif not sync_mode and (episode_stats[7] >= min_reward or episode_stats[8] >= min_avg_reward):
                trainer.save_model = f'models/{settings.MODEL_NAME}_sync{sync_mode}_{episode_stats[8]:_>7.2f}avgW_{episode_stats[9]:_>7.2f}maxW_{episode_stats[7]:_>7.2f}minW_{agent_episode}episode_{int(time.time())}timestamp.model'
                min_reward = episode_stats[7]
                min_avg_reward = episode_stats[8]
                
        time.sleep(0.01)

    trainer_thread.join()

    trainer_stats[0] = TRAINER_STATE.finished