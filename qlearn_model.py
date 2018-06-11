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

from tqdm import tqdm, trange
from Agent import Agent
from RLAgents import DQNAgent
from Agent import rangefloat
from itertools import count


import json
import matplotlib.pyplot as plt

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200 # timesteps to observe before training
EXPLORE = 3000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 5000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 3
LEARNING_RATE = 1e-4

NUM_OF_EPISODES = 800

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def gameInit(game_state, agent):
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    x_t = x_t / 255.0

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4
    agent.initialize(s_t.astype(np.float16), do_nothing.astype(np.uint8))

def runEpisode(game_state, agent, training=False):
    gameInit(game_state, agent)
    steps = count()
    episode_reward = 0
    for t in steps:
        a_t = agent.choose(Training=training)

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        episode_reward += r_t
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1.reshape((1, 80, 80, 1))

        x_t1 = x_t1 / 255.0
        
        agent.feedback(x_t1.astype(np.float16), r_t, terminal, Training=training)
        
        if training is True:
            loss = agent.train()
            
        if terminal is True:
            break
    
    number_of_steps = next(steps) - 1
    
    return episode_reward, number_of_steps

def fillMemory(game_state, agent, frames=2**10):
    gameInit(game_state, agent)
    frames = trange(frames)
    frames.set_description('Filling Memory...')
    for t in frames:
        #choose an action epsilon greedy
        a_t = np.zeros(2)
        a_t[np.random.randint(2)] = 1
        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1.reshape((1, 80, 80, 1))

        x_t1 = x_t1 / 255.0
        agent.feedback(x_t1.astype(np.float16), r_t, terminal)
            
    while terminal is True:
        #choose an action epsilon greedy
        a_t = np.zeros(2)
        a_t[0] = 1

        #run the selected action and observed next state and reward
        _, _, terminal = game_state.frame_step(a_t)
    
def trainAgentEpisodic(agent):
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    #agent = agent_class('model.h5', memory_size=REPLAY_MEMORY,
    #              Epsilon=rangefloat(INITIAL_EPSILON,FINAL_EPSILON,EXPLORE),
    #              K=FRAME_PER_ACTION)

    fillMemory(game_state, agent, OBSERVATION)
    
    episode_scores = []
    with trange(0, NUM_OF_EPISODES) as episodes:
        episodes.set_description('Training...')
        steps = 0
        for episode in episodes:
            _, stp = runEpisode(game_state, agent, training=True)
            steps += stp
            if (episode+1)%20 == 0:
                episodes.set_description('Testing...')
                score, _ = runEpisode(game_state, agent, training=False)
                episode_scores.append(score)
                episodes.set_description('Reward {:.2f} | Epsilon: {:.6f} | Steps {!s} | Training...'.format(np.mean(episode_scores), agent.get_epsilon(), steps))

    print('\nMean: {:.3f} Std: {:.3}'.format(np.mean(episode_scores), np.std(episode_scores)))
    plt.plot(range(0, NUM_OF_EPISODES, 20), episode_scores, 'ro')
    plt.ylabel('Score')
    plt.show()
    print("Episode finished!")
    print("************************")

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--model', help='Model', required=True)
    args = vars(parser.parse_args())
    print(args)
    model = args['model']
    if model == 'DQN':
        agent = Agent('model.h5', memory_size=REPLAY_MEMORY,
                  Epsilon=rangefloat(INITIAL_EPSILON,FINAL_EPSILON,EXPLORE),
                  K=FRAME_PER_ACTION)
        trainAgentEpisodic(agent)
    elif model == 'DDQN':
        agent = DQNAgent('model.h5', memory_size=REPLAY_MEMORY,
                  epsilon=rangefloat(INITIAL_EPSILON,FINAL_EPSILON,EXPLORE),
                  K=FRAME_PER_ACTION, with_target=False, replayType='prioritized')
        trainAgentEpisodic(agent)
    elif model == 'tensorforce':
        with open('cnn_dqn_network.json', 'r') as fp:
            network_spec = json.load(fp=fp)
        agent = PPOAgent(
            states=dict(type='float', shape=(80, 80, 4)),
            actions=dict(type='int', num_actions=2),
            network=network_spec,
            batching_capacity=32,
            step_optimizer=dict(
                type='adam',
                learning_rate=LEARNING_RATE
            )
        )  
    else:
        print('No such model has been implemented.')

if __name__ == "__main__":
    main()
