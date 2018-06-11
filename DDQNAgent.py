import numpy as np
import random

from rangefloat import rangefloat
from NNets import FFNetAsync, FFNetEnsemble
from ReplayMemory import PrioritizedReplay, SimpleReplay
        
class DDQNAgent(object):
    '''An agent class used in testing reinforcement learning algorithms.

    This class is made with the purpose that it would allow multiple agents to
    be trained concurrently in a single game so the majority of their
    work should be hidden behind this class.
    '''

    def __init__(self, model, memory_size=1024,
                Batch_size=32, Gamma=0.99, Epsilon=rangefloat(1.0,0.1,1e6),
                K=1, name='Agent'):
        '''Create Agent from model description file.'''
        self.Memory = PrioritizedReplay(memory_size)
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
        self.model = FFNetAsync(model)
        self.target_model = FFNetAsync(model)
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
                self.model.predict_on_batch(self.current_state)
                x = self.model.collect()
                index = np.argmax(x)
                action[index] = 1
            self.current_action = action.astype(np.uint8)
        return self.current_action
        
    def chooseOptimal(self):
        action = np.zeros(self.current_action.shape)
        self.target_model.predict_on_batch(self.current_state)
        x = self.target_model.collect()
        index = np.argmax(x)
        action[index] = 1
        return action
    
    def feedback(self, frame, reward, terminal):
        '''Receive feedback from Game.'''
        self.model.predict_on_batch(self.current_state)
        new_state = np.append(frame, self.current_state[...,0:-1], axis=3)
        self.target_model.predict_on_batch(new_state)
        
        Q = np.max(self.model.collect().flatten()*self.current_action)
        out = self.target_model.collect().flatten()
        T  = reward + self.Gamma*np.max(out)
        
        self.Memory.insert((self.current_state, self.current_action, reward, new_state, terminal), abs(Q-T))
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
        batch = self.Memory.batch()

        pseq_batch = np.concatenate([b[0] for b in batch], axis=0)
        action_batch = np.stack([b[1] for b in batch])
        reward_batch = np.array([b[2] for b in batch])
        seq_batch = np.concatenate([b[3] for b in batch], axis=0)
        term_batch = np.array([b[4] for b in batch])
        
        self.target_model.predict_on_batch(seq_batch)
        #self.model.predict_on_batch(seq_batch)
        self.model.predict_on_batch(pseq_batch)
        
        out = self.target_model.collect()
        #actions = self.model.collect()
        #out = out[np.arange(len(out)), np.argmax(actions, axis=1)].reshape(-1, 1)
        y_batch = self.model.collect()
        
        y_batch[action_batch==1] = reward_batch  + self.Gamma*np.max(out, axis=1)*np.invert(term_batch)
        self.model.train_on_batch(pseq_batch, y_batch)
        
        self.model.get_weights()
        self.target_model.get_weights()
        
        weights = self.model.collect()
        target_weights = self.target_model.collect()
        for i in range(len(target_weights)):
            target_weights[i] = target_weights[i]*(0.8) + weights[i]*0.2
        self.target_model.set_weights(target_weights)
        
    def get_epsilon(self):
        return self.Epsilon