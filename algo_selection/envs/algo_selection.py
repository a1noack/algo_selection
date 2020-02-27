#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the TA1 algorithm selection environment.

The agent sequentially selects which TA1 algorithms 
to run while taking into consideration accuracy of 
algorithms and the amount of time remaining.
"""

import random
from gym import spaces
import gym
import numpy as np
import pandas as pd
import statistics as st
import math
import time

class AlgoSelection(gym.Env):
    def __init__(self):
        self.timing = pd.read_pickle('env_specs/analytic_timings.pkl')
        self.predictions = pd.read_pickle('env_specs/prediction_data.pkl')
        self.analytic_names = list(self.timing['analytic'])
        self.analytic_times = np.array(self.timing['avg_proc'])
        
        self.num_algos = len(self.analytic_names)
        self.curr_state = np.zeros(self.num_algos)
        self.budget = 1000 # time in seconds
        self.time = 0
        self.preds = []
        self.index = 6
        self.episode_done = False
        self.curr_episode = 0
        self.stdev_thresh = .35
        
    def _algo_used(self, algo):
        return self.curr_state[algo] == 1.
    
    def random_action(self):
        unused = np.where(self.curr_state == 0.)[0]
        num_unused = np.size(unused)
        i = np.random.randint(num_unused)
        return unused[i]
    
    def unused_algos(self):  
        return not np.sum(self.curr_state) == np.size(self.curr_state)

    def step(self, algo, verbose=False):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self._algo_used(algo):
            return self.curr_state, 0., False, {}
        
        self.curr_state[algo] = 1.
        
        if self.time + self.analytic_times[algo] >= self.budget:
            return self.curr_state, 0., False, {}
        
        self.time += self.analytic_times[algo]      
        algo_name = self.analytic_names[algo]
        new_pred = float(self.predictions.at[self.index, algo_name + '_score'])
        
        reward = 0
        if not math.isnan(new_pred):
            self.preds.append(new_pred)
            reward = 1 - (new_pred - float(self.predictions.at[self.index, 'label']))**2
        
        if len(self.preds) > 1:
            stdev = st.stdev(self.preds)
            if stdev < self.stdev_thresh:
                self.episode_done = True
   
        if not self.unused_algos():
            self.episode_done = True
            
        if verbose:
            print('running algo #{}, {}'.format(algo, algo_name))
            print('\ttime to run algo: {}s'.format(self.analytic_times[algo]))      
            print("\talgo prediction: {:.3f}".format(new_pred))           

        return self.curr_state, reward, self.episode_done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.preds = []
        self.time = 0
        if self.index >= self.predictions.shape[0] - 1:
            self.index = 0
        else:
            self.index = self.index + 1
        self.episode_done = False
        self.curr_episode += 1
        self.curr_state = np.zeros(self.num_algos)
        
        return self.curr_state

    def _render(self, mode='human', close=False):
        return