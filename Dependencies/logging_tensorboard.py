import os
import sys
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from keras.callbacks import Callback

# Custom TensorBoard Class (one log file for all .fit() calls)
class TensorB(Callback):

    def __init__(self, log_dir):
        self.step = 1
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    
    def log_init(self, prefix, **params):
        with self.writer.as_default():
            for name, value in params.items():
                if isinstance(value, (int, float)):
                    tf.summary.scalar(f"{prefix}/{name}", value, step=0)
                elif isinstance(value, str):
                    tf.summary.text(f"{prefix}/{name}", tf.constant(value), step=0)
                elif isinstance(value, list):
                    list_str = ', '.join(map(str, value))
                    tf.summary.text(f"{prefix}/{name}", tf.constant(list_str), step=0)
                elif isinstance(value, dict):
                    dict_str = ', '.join(f"{k}: {v}" for k, v in value.items())
                    tf.summary.text(f"{prefix}/{name}", tf.constant(dict_str), step=0)
                else:
                    print(f"Skipping unsupported sensor setting: {name}={value}")
            self.writer.flush()

                
    # Saves logs with our step number (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats('trainer', self.step, **logs)

    def update_stats(self, prefix, step, **stats):
        self._write_logs(prefix, stats, step)

    def _write_logs(self, prefix, logs, step):
        with self.writer.as_default():
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                name = f"{prefix}/{name}"
                tf.summary.scalar(name, value, step=step)
            self.writer.flush()