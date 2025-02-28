#!/usr/bin/env python

#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
'''
This code functionality is a modified version based on the Carla Code.
'''

import settings
import numpy
from simple_pid import PID  
from Dependencies import control_physics as phys

class MPCToCarla():
    """Converts high level Actions to Carla Vehicle Inputs."""
    def __init__(self):

        # Create PID controller for velocity and acceleration
        self.speed_controller = PID(Kp=0.05,
                                    Ki=0.,
                                    Kd=0.5,
                                    sample_time=0.05,
                                    output_limits=(-1., 1.))
        self.accel_controller = PID(Kp=0.05,
                                    Ki=0.,
                                    Kd=0.05,
                                    sample_time=0.05,
                                    output_limits=(-1, 1))

      

        self.dt = settings.FIXED_DELTA_SEC

        self.step = 0

        self.vehicle_status = None
        self.info = {
            "target": {
                "steering_angle": 0.0,
                "speed": 0.0,
                "speed_abs": 0.0,
                "accel": 0.0,
                "jerk": 0.0,
            },
            "current": {
                "time_sec": 0.0,
                "speed": 0.0,
                "speed_abs": 0.0,
                "accel": 0.0,
            },
            "status": {
                "status": 'n/a',
                "speed_control_activation_count": 0,
                "speed_control_accel_delta": 0,
                "speed_control_accel_target": 0,
                "accel_control_pedal_delta": 0,
                "accel_control_pedal_target": 0,
                "brake_upper_border": 0,
                "throttle_lower_border": 0
            },
            "output": {
                "throttle": 0.0,
                "brake": 0.0,
                "steer": 0.0,
                "reverse": False,
                "hand_brake": True,
            },
            
        }

       

    def update_vehicle_status(self, vehicle_status):
        self.vehicle_status = vehicle_status


    def update_vehicle_info(self, physics_control):
        """Setup max values for Vehicle based on the blueprint"""
        self.vehicle_info = physics_control
    
        self.info["restrictions"] = {
            "max_steering_angle": phys.get_vehicle_max_steering_angle(
            self.vehicle_info),
            "max_speed": phys.get_vehicle_max_speed(
            self.vehicle_info),
            "max_accel": phys.get_vehicle_max_acceleration(
            self.vehicle_info),
            "max_decel": phys.get_vehicle_max_deceleration(
            self.vehicle_info),
            "min_accel": 1.0,  
            "max_pedal": min(
            phys.get_vehicle_max_acceleration(self.vehicle_info),
            phys.get_vehicle_max_deceleration(self.vehicle_info))
    }

        
    def set_target_values(self, command):
        """set target Values, they come from the calculated optimal MPC Actions."""
        self.set_target_steering_angle(command['steering_angle'])
        self.set_target_speed(command['speed'])
        self.set_target_accel(command['acceleration'])
        self.set_target_jerk(command['jerk'])

    def set_target_steering_angle(self, target_steering_angle):
        """set target sterring angle."""
        self.info['target']['steering_angle'] = -target_steering_angle
        if abs(self.info['target']['steering_angle']) > self.info['restrictions']['max_steering_angle']:
            #self.logerr("Max steering angle reached, clipping value")
            self.info['target']['steering_angle'] = numpy.clip(
                self.info['target']['steering_angle'],
                -self.info['restrictions']['max_steering_angle'],
                self.info['restrictions']['max_steering_angle'])

    def set_target_speed(self, target_speed):
        """Set target speed."""
        if abs(target_speed) > self.info["restrictions"]["max_speed"]:
            #self.logerr("Max speed reached, clipping value")
            self.info["target"]["speed"] = numpy.clip(
                target_speed,
                -self.info["restrictions"]["max_speed"],
                self.info["restrictions"]["max_speed"]
            )
        else:
            self.info["target"]["speed"] = target_speed
        self.info["target"]["speed_abs"] = abs(self.info["target"]["speed"])

    def set_target_accel(self, target_accel):
        """Set target acceleration."""
        epsilon = 0.00001
        # If speed is set to zero, then use max deceleration value
        if self.info["target"]["speed_abs"] < epsilon:
            self.info["target"]["accel"] = -self.info["restrictions"]["max_decel"]
        else:
            self.info["target"]["accel"] = numpy.clip(
                target_accel,
                -self.info["restrictions"]["max_decel"],
                self.info["restrictions"]["max_accel"]
            )

    def set_target_jerk(self, target_jerk):
        """Set target jerk."""
        self.info["target"]["jerk"] = target_jerk

    def vehicle_control_cycle(self):
        """Performs a vehicle control cycle and updates optimal Actions"""
        # perform actual control
        self.control_steering()
        self.control_stop_and_reverse()
        self.run_speed_control_loop()
        self.run_accel_control_loop()
        if not self.info['output']['hand_brake']:
            self.update_drive_vehicle_control_command()


    def control_steering(self):
        """
        Basic steering control
        """
        self.info['output']['steer'] = self.info['target']['steering_angle'] / \
            self.info['restrictions']['max_steering_angle']

    def control_stop_and_reverse(self):
        """
        Handle stopping and switching to reverse gear.
        """
        # Thresholds for determining state
        standing_still_epsilon = 0.1  # Velocity threshold for standing still
        full_stop_epsilon = 0.00001  # Velocity threshold for a full stop

        # Default state for hand brake
        self.info['output']['hand_brake'] = False

        # Check if vehicle is standing still
        if self.info['current']['speed_abs'] < standing_still_epsilon:
            # Allow change of driving direction
            self.info['status']['status'] = "standing"

            if self.info['target']['speed'] < 0:  # Reverse gear
                if not self.info['output']['reverse']:
                    self.info['output']['reverse'] = True
            elif self.info['target']['speed'] > 0:  # Forward gear
                if self.info['output']['reverse']:
                    self.info['output']['reverse'] = False

            # Check for full stop
            if self.info['target']['speed_abs'] < full_stop_epsilon:
                self.info['status']['status'] = "full stop"
                self.info['status']['speed_control_accel_target'] = 0.0
                self.info['status']['accel_control_pedal_target'] = 0.0
                self.set_target_speed(0.0)
                self.info['current']['speed'] = 0.0
                self.info['current']['speed_abs'] = 0.0
                self.info['current']['accel'] = 0.0
                self.info['output']['hand_brake'] = True
                self.info['output']['brake'] = 1.0
                self.info['output']['throttle'] = 0.0

        # Handle request for direction change while moving
        elif numpy.sign(self.info['current']['speed']) * numpy.sign(self.info['target']['speed']) == -1:
            self.set_target_speed(0.0)

    def run_speed_control_loop(self):
        """
        Run the PID control loop for speed.

        The speed control loop is activated only when the desired acceleration is moderate.
        Otherwise, the controller directly follows the desired acceleration values.

        Reasoning:
        - Autonomous vehicles calculate a trajectory with positions and velocities.
        - The Ackermann drive reflects the desired speed profile.
        - The PID controller is primarily responsible for maintaining speed when there are no significant changes.
        """
        epsilon = 0.00001  # Threshold to handle numerical noise
        target_accel_abs = abs(self.info['target']['accel'])

        # Manage speed control activation count
        if target_accel_abs < self.info['restrictions']['min_accel']:
            # Increment activation count if acceleration is low
            if self.info['status']['speed_control_activation_count'] < 5:
                self.info['status']['speed_control_activation_count'] += 1
        else:
            # Decrement activation count if acceleration is high
            if self.info['status']['speed_control_activation_count'] > 0:
                self.info['status']['speed_control_activation_count'] -= 1

        # Update the auto mode of the speed controller
        self.speed_controller.auto_mode = self.info['status']['speed_control_activation_count'] >= 5

        if self.speed_controller.auto_mode:
            # PID controller active: Maintain target speed
            self.speed_controller.setpoint = self.info['target']['speed_abs']
            self.info['status']['speed_control_accel_delta'] = float(
                self.speed_controller(self.info['current']['speed_abs'])
            )

            # Set acceleration clipping borders
            clipping_lower_border = -target_accel_abs
            clipping_upper_border = target_accel_abs

            # Use max deceleration/acceleration if acceleration is close to zero
            if target_accel_abs < epsilon:
                clipping_lower_border = -self.info['restrictions']['max_decel']
                clipping_upper_border = self.info['restrictions']['max_accel']

            # Clip the target acceleration to stay within bounds
            self.info['status']['speed_control_accel_target'] = numpy.clip(
                self.info['status']['speed_control_accel_target'] +
                self.info['status']['speed_control_accel_delta'],
                clipping_lower_border, clipping_upper_border
            )
        else:
            # PID controller inactive: Follow the desired acceleration directly
            self.info['status']['speed_control_accel_delta'] = 0.0
            self.info['status']['speed_control_accel_target'] = self.info['target']['accel']


    def run_accel_control_loop(self):
        """Run the PID control loop for the acceleration"""
        # setpoint of the acceleration controller is the output of the speed controller
        self.accel_controller.setpoint = self.info['status']['speed_control_accel_target']
        self.info['status']['accel_control_pedal_delta'] = float(self.accel_controller(
            self.info['current']['accel']))
        # @todo: we might want to scale by making use of the the abs-jerk value
        # If the jerk input is big, then the trajectory input expects already quick changes
        # in the acceleration; to respect this we put an additional proportional factor on top
        self.info['status']['accel_control_pedal_target'] = numpy.clip(
            self.info['status']['accel_control_pedal_target'] +
            self.info['status']['accel_control_pedal_delta'],
            -self.info['restrictions']['max_pedal'], self.info['restrictions']['max_pedal'])

    def update_drive_vehicle_control_command(self):
        """
        Apply the current speed_control_target value to throttle/brake commands
        """

        # the driving impedance moves the 'zero' acceleration border
        # Interpretation: To reach a zero acceleration the throttle has to pushed
        # down for a certain amount
        self.info['status']['throttle_lower_border'] = phys.get_vehicle_driving_impedance_acceleration(
            self.vehicle_info, self.vehicle_status, self.info['output']['reverse'])

        # the engine lay off acceleration defines the size of the coasting area
        # Interpretation: The engine already prforms braking on its own;
        #  therefore pushing the brake is not required for small decelerations
        self.info['status']['brake_upper_border'] = self.info['status']['throttle_lower_border'] + \
            phys.get_vehicle_lay_off_engine_acceleration(self.vehicle_info)

        if self.info['status']['accel_control_pedal_target'] > self.info['status']['throttle_lower_border']:
            self.info['status']['status'] = "accelerating"
            self.info['output']['brake'] = 0.0
            # the value has to be normed to max_pedal
            # be aware: is not required to take throttle_lower_border into the scaling factor,
            # because that border is in reality a shift of the coordinate system
            # the global maximum acceleration can practically not be reached anymore because of
            # driving impedance
            self.info['output']['throttle'] = (
                (self.info['status']['accel_control_pedal_target'] -
                 self.info['status']['throttle_lower_border']) /
                abs(self.info['restrictions']['max_pedal']))
        elif self.info['status']['accel_control_pedal_target'] > self.info['status']['brake_upper_border']:
            self.info['status']['status'] = "coasting"
            # no control required
            self.info['output']['brake'] = 0.0
            self.info['output']['throttle'] = 0.0
        else:
            self.info['status']['status'] = "braking"
            # braking required
            self.info['output']['brake'] = (
                (self.info['status']['brake_upper_border'] -
                 self.info['status']['accel_control_pedal_target']) /
                abs(self.info['restrictions']['max_pedal']))
            self.info['output']['throttle'] = 0.0

        # finally clip the final control output (should actually never happen)
        self.info['output']['brake'] = numpy.clip(
            self.info['output']['brake'], 0., 1.)
        self.info['output']['throttle'] = numpy.clip(
            self.info['output']['throttle'], 0., 1.)


    def update_current_values(self, velocity):
        """Updates the current Vehicle State"""
        delta_time = self.dt if self.step > 0 else 0
        current_speed = velocity
        if delta_time > 0:
            delta_speed = current_speed - self.info['current']['speed']
            current_accel = delta_speed / delta_time
            # average filter
            self.info['current']['accel'] = (self.info['current']['accel'] * 4 + current_accel) / 5
        self.info['current']['speed'] = current_speed
        self.info['current']['speed_abs'] = abs(current_speed)


    def get_output(self):
        return self.info['output']
    
