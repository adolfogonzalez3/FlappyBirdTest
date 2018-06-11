#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

from tqdm import tqdm
from Agent import Agent as Agent
from Agent import rangefloat
from itertools import count

import matplotlib.pyplot as plt

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 5000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 3
LEARNING_RATE = 1e-4

NUM_OF_EPISODES = 200#800

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def trainAgent():
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        
class Game(object):
    
    def __init__(self):
        

class FFNetFunctions(Enum):
    predict_on_batch = auto()
    train_on_batch = auto()

class Message(object):
    def __init__(self, header, data):
        self.header = header
        self.data = data

def net_async(pipe, args, kwargs):
    mydata = threading.local()
    mydata.net = FeedForwardNet(*args, **kwargs)
    while True:
        mydata.message = pipe[0].get()
        data = mydata.message.data
        if mydata.message.header == FFNetFunctions.predict_on_batch:
            pipe[1].put(mydata.net.predict_on_batch(data))
        elif mydata.message.header == FFNetFunctions.train_on_batch:
            loss = mydata.net.train_on_batch(data[0], data[1])