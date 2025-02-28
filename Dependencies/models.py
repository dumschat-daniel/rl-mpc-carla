import os
import settings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten, GlobalMaxPooling1D, Conv1D, BatchNormalization, LayerNormalization

MODEL_NAME_PREFIX = ''





def model_head_dqn(state_encoder, output_size):
    x = state_encoder.output

    x = Dense(64)(x)
    if settings.MODEL_NORMALIZATION == 'batch':
        x = BatchNormalization()(x)  
    elif settings.MODEL_NORMALIZATION == 'layer':
        x = LayerNormalization()(x)
        
    x = Activation('relu')(x)  
    outputs = Dense(output_size, activation='linear', name='distinct_actions')(x)

    model = Model(inputs=state_encoder.inputs, outputs=outputs)

    return model


def actor_head_ddpg(state_encoder, output_size):
    x = state_encoder.output

    x = Dense(128, activation="relu")(x)
    if settings.MODEL_NORMALIZATION == 'batch':
        x = BatchNormalization()(x)  
    elif settings.MODEL_NORMALIZATION == 'layer':
        x = LayerNormalization()(x)

    x = Dense(128, activation="relu")(x)
    if settings.MODEL_NORMALIZATION == 'batch':
        x = BatchNormalization()(x)  
    elif settings.MODEL_NORMALIZATION == 'layer':
        x = LayerNormalization()(x)

    x = Dense(64, activation="relu")(x)
    if settings.MODEL_NORMALIZATION == 'batch':
        x = BatchNormalization()(x)  
    elif settings.MODEL_NORMALIZATION == 'layer':
        x = LayerNormalization()(x)

    outputs = Dense(output_size, activation='tanh', name='outputs')(x)  
    model = Model(inputs=state_encoder.inputs, outputs=outputs)
    return model


def critic_head_ddpg(state_encoder, action_dim):
    x = state_encoder.output
    action_input = Input(shape=(action_dim,), name='action_input')

    if settings.DDPG_PREPROCESS_ACTION_INPUT:
        action_output = Dense(256, activation='relu')(action_input)  
    else:
        action_output = action_input
    x = Concatenate()([x, action_output])

    x = Dense(128, activation="relu")(x)
    if settings.MODEL_NORMALIZATION == 'batch':
        x = BatchNormalization()(x)  
    elif settings.MODEL_NORMALIZATION == 'layer':
        x = LayerNormalization()(x)
    x = Dense(128, activation="relu")(x)
    if settings.MODEL_NORMALIZATION == 'batch':
        x = BatchNormalization()(x)  
    elif settings.MODEL_NORMALIZATION == 'layer':
        x = LayerNormalization()(x)
    x = Dense(64, activation="relu")(x)
    if settings.MODEL_NORMALIZATION == 'batch':
        x = BatchNormalization()(x)  
    elif settings.MODEL_NORMALIZATION == 'layer':
        x = LayerNormalization()(x)
    
    outputs = Dense(1)(x)

    model = Model(inputs=[state_encoder.inputs, action_input], outputs=outputs)
    return model



# Base class. Gets a head attached to it (dqn, ddpg, td3)
def state_encoder(model_inputs, model_settings):
    # Initialize inputs and x
    inputs = []
    x = None

    # Handle model_inputs if they exist
    if model_inputs:
        inputs.extend([model.input for model in model_inputs])
        model_outputs = [model.output for model in model_inputs]
        x = Concatenate()(model_outputs) if len(model_outputs) > 1 else model_outputs[0]

    # Additional inputs like GNSS, collision, speed, etc.
    if model_settings['relative_pos']:
        navigation_input = Input(shape=(3 if model_settings['relative_orientation'] else 2,), name='navigation_input')
        y = Dense(64, activation='relu')(navigation_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(navigation_input)

    if model_settings['speed']:
        speed_input = Input(shape=(1,), name='speed_input')
        y = Dense(32, activation='relu')(speed_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(speed_input)
        
    if model_settings['collision']:
        collision_input = Input(shape=(1,), name='collision_input')
        y = Dense(32, activation='relu')(collision_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(collision_input)    

    if model_settings['lane_invasion']:
        lane_invasion_input = Input(shape=(1,), name='lane_invasion_input')
        y = Dense(32, activation='relu')(lane_invasion_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(lane_invasion_input)

    if model_settings['last_action']:
        last_action_input = Input(shape=(1,), name='last_action_input')
        y = Dense(32, activation='relu')(last_action_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(last_action_input)
    
    if model_settings['last_agent_input']:
        last_agent_input_input = Input(shape=(1,), name='last_agent_input_input')
        y = Dense(32, activation='relu')(last_agent_input_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(last_agent_input_input)

    if model_settings['distance_to_lane_center']:
        lane_center_input = Input(shape=(3 if model_settings['orientation_difference_to_lane_center'] else 2,), name='lane_center_input')
        y = Dense(32, activation='relu')(lane_center_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(lane_center_input) 

    if model_settings['acceleration']:
        acceleration_input = Input(shape=(1,), name='acceleration_input')
        y = Dense(32, activation='relu')(acceleration_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(acceleration_input)  

    if model_settings['yaw_angle']:
        yaw_angle_input = Input(shape=(1,), name='yaw_angle_input')
        y = Dense(32, activation='relu')(yaw_angle_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(yaw_angle_input)  

    if model_settings['jerk_rate']:
        yaw_rate_input = Input(shape=(1,), name='yaw_rate_input')
        y = Dense(32, activation='relu')(yaw_rate_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(yaw_rate_input)

    if model_settings['traffic_light_state']:
        traffic_light_state_input = Input(shape=(1,), name='traffic_light_state_input')
        y = Dense(32, activation='relu')(traffic_light_state_input)
        x = y if x is None else Concatenate()([x, y])
        inputs.append(traffic_light_state_input)  


    return Model(inputs=inputs, outputs=x)



# --------------------------------------------------------------------
# Models with camera only. 

def model_base_Xception(input_shape):
    model = Xception(weights=None, include_top=False, input_shape=input_shape)
    x = model.output
    x = GlobalAveragePooling2D()(x)

    return model.input, x

def model_base_test_CNN(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())

    return model.input, model.output

def model_base_64x3_CNN(input_shape):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Flatten())

    return model.input, model.output

def model_base_4_CNN(input_shape):
    model = Sequential()

    model.add(Conv2D(64, (5, 5), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())

    return model.input, model.output

def model_base_5_residual_CNN(input_shape):
    input = Input(shape=input_shape)

    cnn_1 = Conv2D(64, (7, 7), padding='same')(input)
    cnn_1a = Activation('relu')(cnn_1)
    cnn_1c = Concatenate()([cnn_1a, input])
    cnn_1ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1c)

    cnn_2 = Conv2D(64, (5, 5), padding='same')(cnn_1ap)
    cnn_2a = Activation('relu')(cnn_2)
    cnn_2c = Concatenate()([cnn_2a, cnn_1ap])
    cnn_2ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2c)

    cnn_3 = Conv2D(128, (5, 5), padding='same')(cnn_2ap)
    cnn_3a = Activation('relu')(cnn_3)
    cnn_3c = Concatenate()([cnn_3a, cnn_2ap])
    cnn_3ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_3c)

    cnn_4 = Conv2D(256, (5, 5), padding='same')(cnn_3ap)
    cnn_4a = Activation('relu')(cnn_4)
    cnn_4c = Concatenate()([cnn_4a, cnn_3ap])
    cnn_4ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_4c)

    cnn_5 = Conv2D(512, (3, 3), padding='same')(cnn_4ap)
    cnn_5a = Activation('relu')(cnn_5)
    #cnn_5c = Concatenate()([cnn_5a, cnn_4ap])
    cnn_5ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_5a)

    flatten = Flatten()(cnn_5ap)

    return input, flatten

def model_base_5_residual_CNN_noact(input_shape):
    input = Input(shape=input_shape)

    cnn_1 = Conv2D(64, (7, 7), padding='same')(input)
    #cnn_1a = Activation('relu')(cnn_1)
    cnn_1c = Concatenate()([cnn_1, input])
    cnn_1ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1c)

    cnn_2 = Conv2D(64, (5, 5), padding='same')(cnn_1ap)
    #cnn_2a = Activation('relu')(cnn_2)
    cnn_2c = Concatenate()([cnn_2, cnn_1ap])
    cnn_2ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2c)

    cnn_3 = Conv2D(128, (5, 5), padding='same')(cnn_2ap)
    #cnn_3a = Activation('relu')(cnn_3)
    cnn_3c = Concatenate()([cnn_3, cnn_2ap])
    cnn_3ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_3c)

    cnn_4 = Conv2D(256, (5, 5), padding='same')(cnn_3ap)
    #cnn_4a = Activation('relu')(cnn_4)
    cnn_4c = Concatenate()([cnn_4, cnn_3ap])
    cnn_4ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_4c)

    cnn_5 = Conv2D(512, (3, 3), padding='same')(cnn_4ap)
    #cnn_5a = Activation('relu')(cnn_5)
    #cnn_5c = Concatenate()([cnn_5a, cnn_4ap])
    cnn_5ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_5)

    flatten = Flatten()(cnn_5ap)

    return input, flatten

# 5 CNN layer with residual connections model
def model_base_5_wide_CNN(input_shape):
    input = Input(shape=input_shape)

    cnn_1_c1 = Conv2D(64, (7, 7), strides=(3, 3), padding='same')(input)
    cnn_1_a = Activation('relu')(cnn_1_c1)

    cnn_2_c1 = Conv2D(64, (5, 5), strides=(3, 3), padding='same')(cnn_1_a)
    cnn_2_a1 = Activation('relu')(cnn_2_c1)
    cnn_2_c2 = Conv2D(64, (3, 3), strides=(3, 3), padding='same')(cnn_1_a)
    cnn_2_a2 = Activation('relu')(cnn_2_c2)
    cnn_2_ap = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(cnn_1_a)
    cnn_2_c = Concatenate()([cnn_2_a1, cnn_2_a2, cnn_2_ap])

    cnn_3_c1 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(cnn_2_c)
    cnn_3_a1 = Activation('relu')(cnn_3_c1)
    cnn_3_c2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(cnn_2_c)
    cnn_3_a2 = Activation('relu')(cnn_3_c2)
    cnn_3_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_2_c)
    cnn_3_c = Concatenate()([cnn_3_a1, cnn_3_a2, cnn_3_ap])

    cnn_4_c1 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(cnn_3_c)
    cnn_4_a1 = Activation('relu')(cnn_4_c1)
    cnn_4_c2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(cnn_3_c)
    cnn_4_a2 = Activation('relu')(cnn_4_c2)
    cnn_4_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_3_c)
    cnn_4_c = Concatenate()([cnn_4_a1, cnn_4_a2, cnn_4_ap])

    cnn_5_c1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(cnn_4_c)
    cnn_5_a1 = Activation('relu')(cnn_5_c1)
    cnn_5_gap = GlobalAveragePooling2D()(cnn_5_a1)

    return input, cnn_5_gap

def model_base_5_wide_CNN_noact(input_shape):
    input = Input(shape=input_shape)

    cnn_1_c1 = Conv2D(64, (7, 7), strides=(3, 3), padding='same')(input)
    cnn_1_a = Activation('relu')(cnn_1_c1)

    cnn_2_c1 = Conv2D(64, (5, 5), strides=(3, 3), padding='same')(cnn_1_a)
    #cnn_2_a1 = Activation('relu')(cnn_2_c1)
    cnn_2_c2 = Conv2D(64, (3, 3), strides=(3, 3), padding='same')(cnn_1_a)
    #cnn_2_a2 = Activation('relu')(cnn_2_c2)
    cnn_2_ap = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(cnn_1_a)
    cnn_2_c = Concatenate()([cnn_2_c1, cnn_2_c2, cnn_2_ap])

    cnn_3_c1 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(cnn_2_c)
    #cnn_3_a1 = Activation('relu')(cnn_3_c1)
    cnn_3_c2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(cnn_2_c)
    #cnn_3_a2 = Activation('relu')(cnn_3_c2)
    cnn_3_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_2_c)
    cnn_3_c = Concatenate()([cnn_3_c1, cnn_3_c2, cnn_3_ap])

    cnn_4_c1 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(cnn_3_c)
    #cnn_4_a1 = Activation('relu')(cnn_4_c1)
    cnn_4_c2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(cnn_3_c)
    #cnn_4_a2 = Activation('relu')(cnn_4_c2)
    cnn_4_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_3_c)
    cnn_4_c = Concatenate()([cnn_4_c1, cnn_4_c2, cnn_4_ap])

    cnn_5_c1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(cnn_4_c)
    #cnn_5_a1 = Activation('relu')(cnn_5_c1)
    cnn_5_gap = GlobalAveragePooling2D()(cnn_5_c1)

    return input, cnn_5_gap


def model_base_resnet50(input_shape, sensor_settings, base_name="resnet"):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add unique name prefix to all layers to avoid name conflicts
    for layer in base_model.layers:
        layer._name = base_name + "_" + layer.name

    x = GlobalAveragePooling2D(name=base_name + "_avg_pool")(base_model.output)

    num_of_dense_layers = sensor_settings.get('num_of_dense_layers', 2)
    num_of_outputs = sensor_settings.get('num_of_outputs', 128)
    current_outputs = num_of_outputs

    for i in range(num_of_dense_layers - 1):
        current_outputs *= 2
        x = Dense(current_outputs, activation='relu', name=f"{base_name}_dense_{i}")(x)

    while current_outputs >= num_of_outputs:
        x = Dense(current_outputs, activation='relu', name=f"{base_name}_dense_final_{current_outputs}")(x)
        current_outputs //= 2

    return Model(inputs=base_model.input, outputs=x, name=base_name + "_model")


def model_base_pointnet(input_shape):
    input = Input(shape=input_shape, name='lidar_input')

    x = Conv1D(64, kernel_size=1, activation='relu')(input)
    x = Conv1D(128, kernel_size=1, activation='relu')(x)
    x = Conv1D(1024, kernel_size=1, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    return Model(inputs=input, outputs=x)

# --------------------------------------------------------------------------------






