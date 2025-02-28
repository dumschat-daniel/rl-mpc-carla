import settings
import numpy as np
from abc import ABC, abstractmethod
import copy

class NoiseInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def sample(self):
        pass


class GaussianNoise(NoiseInterface):
    """Simple Gaussian Noise implementation with Noise Decay."""
    def __init__(self):
        # copy because of multi agent
        self.throttle_params = copy.deepcopy(settings.NOISE_THROTTLE_PARAMS)
        self.steering_params = copy.deepcopy(settings.NOISE_STEERING_PARAMS)
        
        self.initial_throttle_sigma = self.throttle_params['sigma']
        self.initial_steering_sigma = self.steering_params['sigma']

        self.reset_count = 0

    def reset(self):
        """Apply decay strategy based on settings."""
        self.reset_count += 1  

        if settings.NOISE_DECAY_STRATEGY == "exponential":
            self.throttle_params['sigma'] = max(
                self.throttle_params['min_sigma'],
                self.throttle_params['sigma'] * self.throttle_params['sigma_decay']
            )
            self.steering_params['sigma'] = max(
                self.steering_params['min_sigma'],
                self.steering_params['sigma'] * self.steering_params['sigma_decay']
            )

        elif settings.NOISE_DECAY_STRATEGY == "linear":
            # Decay = original_sigma * sigma_decay * step count
            linear_decay_throttle = self.initial_throttle_sigma * self.throttle_params['sigma_decay'] * self.reset_count
            linear_decay_steering = self.initial_steering_sigma * self.steering_params['sigma_decay'] * self.reset_count
            
            self.throttle_params['sigma'] = max(
                self.throttle_params['min_sigma'],
                self.initial_throttle_sigma - linear_decay_throttle
            )
            self.steering_params['sigma'] = max(
                self.steering_params['min_sigma'],
                self.initial_steering_sigma - linear_decay_steering
            )

        elif settings.NOISE_DECAY_STRATEGY == "logarithmic":
            self.throttle_params['sigma'] = max(
                self.throttle_params['min_sigma'],
                self.throttle_params['min_sigma'] + (self.initial_throttle_sigma - self.throttle_params['min_sigma']) / np.log(self.reset_count + 1)
            )
            self.steering_params['sigma'] = max(
                self.steering_params['min_sigma'],
                self.steering_params['min_sigma'] + (self.initial_steering_sigma - self.steering_params['min_sigma']) / np.log(self.reset_count + 1)
            )

    def sample(self):
        """Generate Gaussian noise for throttle and steering."""
        throttle_noise = np.random.normal(self.throttle_params['mu'], self.throttle_params['sigma'])
        steering_noise = np.random.normal(self.steering_params['mu'], self.steering_params['sigma'])
        return np.array([throttle_noise, steering_noise])


class OUNoise(NoiseInterface):
    """OUNoise Implementation"""
    
    def __init__(self):
        # copy because of multi agent
        self.throttle_params = copy.deepcopy(settings.NOISE_THROTTLE_PARAMS)
        self.steering_params = copy.deepcopy(settings.NOISE_STEERING_PARAMS)

        self.initial_throttle_sigma = self.throttle_params['sigma']
        self.initial_steering_sigma = self.steering_params['sigma']

        self.state = np.array([self.throttle_params['mu'], self.steering_params['mu']])
        self.reset_count = 0  

    def reset(self):
        """Reset state and apply decay based on settings."""
        self.state = np.array([self.throttle_params['mu'], self.steering_params['mu']])
        self.reset_count += 1

        if settings.NOISE_DECAY_STRATEGY == "exponential":
            self.throttle_params['sigma'] = max(
                self.throttle_params['min_sigma'],
                self.throttle_params['sigma'] * self.throttle_params['sigma_decay']
            )
            self.steering_params['sigma'] = max(
                self.steering_params['min_sigma'],
                self.steering_params['sigma'] * self.steering_params['sigma_decay']
            )

        elif settings.NOISE_DECAY_STRATEGY == "linear":
            linear_decay_throttle = self.initial_throttle_sigma * self.throttle_params['sigma_decay'] * self.reset_count
            linear_decay_steering = self.initial_steering_sigma * self.steering_params['sigma_decay'] * self.reset_count

            self.throttle_params['sigma'] = max(
                self.throttle_params['min_sigma'],
                self.initial_throttle_sigma - linear_decay_throttle
            )
            self.steering_params['sigma'] = max(
                self.steering_params['min_sigma'],
                self.initial_steering_sigma - linear_decay_steering
            )

        elif settings.NOISE_DECAY_STRATEGY == "logarithmic":
            self.throttle_params['sigma'] = max(
                self.throttle_params['min_sigma'],
                self.throttle_params['min_sigma'] + (self.initial_throttle_sigma - self.throttle_params['min_sigma']) / np.log(self.reset_count + 1)
            )
            self.steering_params['sigma'] = max(
                self.steering_params['min_sigma'],
                self.steering_params['min_sigma'] + (self.initial_steering_sigma - self.steering_params['min_sigma']) / np.log(self.reset_count + 1)
            )

    def sample(self):
        """Generate OU process for throttle and steering."""
        x = self.state
        dx_throttle = self.throttle_params['theta'] * (self.throttle_params['mu'] - x[0]) + self.throttle_params['sigma'] * np.random.randn()
        dx_steering = self.steering_params['theta'] * (self.steering_params['mu'] - x[1]) + self.steering_params['sigma'] * np.random.randn()

        self.state = np.array([x[0] + dx_throttle, x[1] + dx_steering])
        
        return self.state
