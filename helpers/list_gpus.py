import os
import sys
import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.python.client import device_lib

# Get list of all devices
devices = device_lib.list_local_devices()

# Print GPUs only
print('\n\n\nList of found GPUs:')
for device in devices:
    if device.device_type == 'GPU':
        print(device.physical_device_desc)