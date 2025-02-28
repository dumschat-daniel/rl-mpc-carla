import settings
import numpy as np
import random
from collections import deque
from abc import ABC, abstractmethod



class ReplayMemoryInterface(ABC):
    """Abstract base class for replay memory."""
    
    @abstractmethod
    def __init__(self, capacity):
        pass

    @abstractmethod
    def add(self, transition):
        pass

    @abstractmethod
    def get_memory_size(self):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

class SumTree:
    """A sum-tree data structure for efficient priority sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree
        self.data = np.zeros(capacity, dtype=object)  # Experiences
        self.size = 0 
        self.ptr = 0  # Pointer for inserting new elements

    def add(self, priority, experience):
        
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = experience
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity # Circular buffer for overwriting old Experiences
        self.size = min(self.size + 1, self.capacity)


    def update(self, idx, priority):
        change = priority - self.tree[idx] # Compute change in priority
        self.tree[idx] = priority
        while idx != 0: # Propagate changes up the tree
            idx = (idx - 1) // 2
            self.tree[idx] += change 

    def get_leaf(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left, right = 2 * idx + 1, 2 * idx + 2
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx, self.tree[idx], self.data[idx - self.capacity + 1] # Return leaf index, priority, and experience

    def total_priority(self):
        return max(self.tree[0], 1e-6)  # Avoid zero priority issues


class PrioritizedReplayMemory(ReplayMemoryInterface):
    """Implements multiple prioritization strategies."""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, lambda_reward=0.5, method='td_error', beta_scale=1e-5):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Prioritization factor
        self.beta = beta  # Importance sampling correction factor
        self.beta_scale = beta_scale # upscaling of beta
        self.epsilon = 1e-5  # Small value to avoid zero priority
        self.lambda_reward = lambda_reward
        self.method = method # Defines how priority is computed
        
    def add(self, transition):
        data, td_error, reducible_loss = transition 
        reward = data[2]

        # add transition and calculate priority
        if self.method == 'td_error':
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        elif self.method == 'reward':
            priority = (abs(reward) + self.epsilon) ** self.alpha
        elif self.method == 'td_error_reducible_loss':
            priority = (abs(td_error) + reducible_loss + self.epsilon) ** self.alpha
        elif self.method == 'td_error_reward':  
            priority = (abs(td_error) + self.lambda_reward * abs(reward) + self.epsilon) ** self.alpha

        else:
            raise ValueError("Unknown prioritization method")
        
        
        self.tree.add(priority, (data, td_error, reducible_loss))
        

    def get_memory_size(self):
        return min(self.tree.size, self.capacity)

    def sample(self, batch_size):
        """Samples a batch based on priority."""
        batch = []
        indices = []
        priorities = []
        total_priority = self.tree.total_priority()
        segment = total_priority / batch_size

        # Gradually increase beta to reduce bias in learning
        self.beta = min(self.beta + self.beta * self.beta_scale, 1)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get_leaf(s)  
            if not isinstance(data, tuple) or len(data) != 3:
                raise TypeError(f"Invalid data structure from leaf: {data} at index {idx}")
            priorities.append(priority)
            indices.append(idx)
            batch.append(data[0]) 

        # Compute importance-sampling weights
        sampling_probs = np.array(priorities) / max(total_priority, 1e-6)
        weights = np.power(len(self.tree.data) * sampling_probs, -self.beta, where=sampling_probs > 0)
        weights /= max(np.max(weights), 1e-6)  # Avoid division by zero

        return batch, indices, weights  



    def update_priorities(self, indices, td_errors=None, rewards=None, reducible_losses=None):
        """Updates priorities based on the selected method."""
        for i, idx in enumerate(indices):
            if self.method == 'td_error':
                priority = (abs(td_errors[i] if td_errors is not None else 0.0) + self.epsilon) ** self.alpha
            elif self.method == 'reward':
                priority = (abs(rewards[i] if rewards is not None else 0.0) + self.epsilon) ** self.alpha
            elif self.method == 'td_error_reducible_loss':
                priority = (abs(td_errors[i] if td_errors is not None else 0.0) + 
                            abs(reducible_losses[i] if reducible_losses is not None else 0.0) + 
                            self.epsilon) ** self.alpha
            elif self.method == 'td_error_reward': 
                priority = (abs(td_errors[i] if td_errors is not None else 0.0) +
                self.lambda_reward * abs(rewards[i] if rewards is not None else 0.0) +
                self.epsilon) ** self.alpha
            else:
                raise ValueError("Unknown prioritization method")
            
            self.tree.update(idx, priority)












class Reward_priorisation_experience_replay(ReplayMemoryInterface):
    """simpler reward based replay memory. Doesn't update priorities and is not based on a tree Structure."""    
    def __init__(self):
        self.main_replay_memory = deque(maxlen=int(settings.EXPERIENCE_REPLAY_SIZE * 0.8))
        self.priority_replay_memory = deque(maxlen=int(settings.EXPERIENCE_REPLAY_SIZE * 0.2))

        self.priority_area = [0,0]
        self.transition_iteration = 0
        self.calculate_priority_area_every = 1000

    def add(self, transition):
        self.transition_iteration += 1
        reward = transition[2]
        if self.is_high_priority(reward):
            if len(self.priority_replay_memory) >= self.priority_replay_memory.maxlen:
                self.main_replay_memory.append(transition)
            else:
                self.priority_replay_memory.append(transition)
        else:
            self.main_replay_memory.append(transition)


        if self.transition_iteration % self.calculate_priority_area_every  == 0:
            self.update_priority_area()
    
    # Priority area uses the upper and lower ends of the reward range (25 and 75 percentile)
    def update_priority_area(self):
       
        all_rewards = [transition[2] for transition in self.main_replay_memory] + \
                      [transition[2] for transition in self.priority_replay_memory]
          
        penalties = [reward for reward in all_rewards if reward < 0]
        rewards_only = [reward for reward in all_rewards if reward > 0]
        

        penalty_25th_percentile = 0
        reward_75th_percentile = 0

        if penalties:
            penalty_25th_percentile = np.percentile(penalties, 25)

        if rewards_only:
            reward_75th_percentile = np.percentile(rewards_only, 75)
        else:
            reward_75th_percentile = 0

        
        self.priority_area = [penalty_25th_percentile, reward_75th_percentile]



    def is_high_priority(self, reward):
        penalty_25th_percentile, reward_75th_percentile = self.priority_area
        return (reward <= penalty_25th_percentile or reward >= reward_75th_percentile)
    


    def get_memory_size(self):
        return len(self.main_replay_memory) + len(self.priority_replay_memory)
    


    def sample(self, batch_size):

        if self.get_memory_size() < batch_size:
            raise ValueError(f"Not enough samples in replay memory to satisfy batch size of {batch_size}.")
        
        priority_count = int(batch_size * 0.5) 


        priority_samples = random.sample(self.priority_replay_memory, 
                                         min(priority_count, len(self.priority_replay_memory)))

        
        remaining_count = batch_size - len(priority_samples)
        main_samples = random.sample(self.main_replay_memory, 
                                     min(remaining_count, len(self.main_replay_memory)))

        
        return priority_samples + main_samples
    

