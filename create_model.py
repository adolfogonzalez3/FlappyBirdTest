import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, Flatten, Reshape, BatchNormalization, Activation
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
import random
from keras import losses
from keras import backend
from collections import deque
from enum import Enum, auto
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

LEARNING_RATE = 1e-4

def func(x):
    from keras import backend
    return backend.sum(x[0]*x[1], axis=1)

def create_model_qlearn():
    frame_inputs = Input(shape=(80,80,4), name='frame_inputs')
    x = BatchNormalization()(frame_inputs)
    x = Conv2D(32, (8, 8), strides=(4,4), padding='same')(x) # 16 by 16 by 4
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (4, 4), strides=(2,2), padding='same')(x) # 8 by 8 by 4
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x) # 8 by 8 by 4
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    Q_value_outputs = Dense(2, activation='linear', name='Output')(x)
    model = Model(inputs=frame_inputs, outputs=Q_value_outputs)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=adam, loss='mse')
    return model

def create_model_starcraft():
    frame_inputs = Input(shape=(64,64,4), name='frame_inputs')
    x = Conv2D(32, (8, 8), strides=(4,4), padding='same')(frame_inputs) # 16 by 16 by 4
    x = Activation('relu')(x)
    x = Conv2D(64, (4, 4), strides=(2,2), padding='same')(x) # 8 by 8 by 4
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x) # 8 by 8 by 4
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Activation('relu')(x)
    Q_value_outputs = Dense(17, activation='linear', name='Output')(x)
    model = Model(inputs=frame_inputs, outputs=Q_value_outputs)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=adam, loss='mse')
    return model
    
def create_model_test():
    net_input = Input(shape=(10,), name='Input')
    x = Dense(5, activation='sigmoid')(net_input)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=net_input, outputs=x)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=adam, loss='mse')
    return model

if __name__ == '__main__':
    model = create_model_starcraft()
    model.save('model_starcraft.h5')

    if False:
        model = create_model_test()
        model.save('model_test.h5')
    
        model = create_model_qlearn()
        model.save('model_qlearn.h5')
    
        
    