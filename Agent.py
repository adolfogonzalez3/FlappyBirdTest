import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
from keras.layers import Input, Dense, Conv2D, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import RMSprop
import random
from keras import losses
from keras import backend
from collections import deque
from enum import Enum, auto
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

from queue import Queue
import threading
from threading import Thread

from rangefloat import rangefloat
from NNets import FFNet, FFNetEnsemble


def create_session(memory_fraction=0.1, allow_growth=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    return tf.Session(config=config)
        
class Agent(object):
    '''An agent class used in testing reinforcement learning algorithms.

    This class is made with the purpose that it would allow multiple agents to
    be trained concurrently in a single game so the majority of their
    work should be hidden behind this class.
    '''

    def __init__(self, model, memory_size=1024,
                Batch_size=32, Gamma=0.99, Epsilon=rangefloat(1.0,0.1,1e6),
                K=1, name='Agent'):
        '''Create Agent from model description file.'''
        self.Memory = deque(maxlen=memory_size)
        self.Batch_size = Batch_size
        self.Gamma = Gamma
        if type(Epsilon) is float or type(Epsilon) is int:
            self.Epsilon = Epsilon
            self.Epsilon_gen = None
        else:
            self.Epsilon_gen = Epsilon
            self.Epsilon = next(self.Epsilon_gen)
        self.K = K
        self.current_state = None
        self.current_action = None
        self.model = FFNet(model)
        self.terminal = False

    def initialize(self, current_state, current_action):
        self.current_state = current_state
        self.current_action = current_action
        self.terminal = False

    def chooseAction(self, time_step):
        '''Choose an action based on the current state.'''
        action = np.zeros(self.current_action.shape)
        if time_step%self.K == 0:
            if random.random() <= self.Epsilon:
                index = [random.randint(0, i-1) for i in action.shape]
                action[index] = 1
            else:
                x = self.model.predict_on_batch(self.current_state)
                index = np.argmax(x)
                action[index] = 1
            self.current_action = action.astype(np.uint8)
        return self.current_action
        
    def chooseOptimal(self):
        action = np.zeros(self.current_action.shape)
        x = self.model.predict_on_batch(self.current_state)
        index = np.argmax(x)
        action[index] = 1
        return action
    
    def feedback(self, frame, reward, terminal):
        '''Receive feedback from Game.'''
        new_state = np.append(frame, self.current_state[...,0:-1], axis=3)
        self.Memory.append((self.current_state, self.current_action, reward, new_state, terminal))
        self.current_state = new_state
        self.terminal = terminal

    def isTerminal(self):
        return self.terminal

    def save(self, name):
        #self.model.save(name)
        pass

    def train(self):
        '''Train the Agent.'''
        if self.Epsilon_gen is not None:
            self.Epsilon = next(self.Epsilon_gen)
        batch = random.sample(self.Memory, self.Batch_size)

        pseq_batch = np.concatenate([b[0] for b in batch], axis=0)
        action_batch = np.stack([b[1] for b in batch])
        reward_batch = np.array([b[2] for b in batch])
        seq_batch = np.concatenate([b[3] for b in batch], axis=0)
        term_batch = np.array([b[4] for b in batch])
        
        
        
        out = self.model.predict_on_batch(seq_batch)
        y_batch = self.model.predict_on_batch(pseq_batch)
        y_batch[action_batch==1] = reward_batch  + self.Gamma*np.max(out, axis=1)*np.invert(term_batch)
        return self.model.train_on_batch(pseq_batch, y_batch)

    def get_epsilon(self):
        return self.Epsilon
        