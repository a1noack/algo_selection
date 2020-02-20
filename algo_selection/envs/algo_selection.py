#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the TA1 algorithm selection environment.

The agent sequentially selects which TA1 algorithms 
to run while taking into consideration accuracy of 
algorithms and the amount of time remaining.
"""

# core modules
#import logging.config
#import math
#import pkg_resources
#import random

from gym import spaces
#import cfg_load
import gym
import numpy as np
import pandas as pd
import statistics as st


class AlgoSelection(gym.Env):
    def __init__(self, data_path, budget, stdev_thresh):
        self._load_datafile()
        self.budget = budget
        self.preds = []
        self.time = 0
        self.index = 0
        self.episode_done = False
        self.curr_episode = 0
        self.stdev_thresh = stdev_thresh
        self.curr_state = np.zeros(self.num_algos)
    
    def _load_datafile(data_path):
        data = pd.read_csv(data_path, sep='\t').values
        self.X, self.Y = data[:, 0], data[:, 1]
        self.algos = data[:, 2:]
        self.num_algos = self.algos.shape[1]

    def step(self, action):
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
        self.current_state[action] = 1.
        self.time += self.algos[self.index, action][1]
        new_pred = self.algos[self.index, action][0]
        preds.append(new_pred)
        if st.stdev(preds) < thresh:
            self.episode_done = True
        
        return self.current_state, reward, self.episode_done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.preds = []
        self.time = 0
        self.index = self.index + 1
        self.episode_done = False
        self.curr_episode += 1
        self.curr_state = np.zeros(self.num_algos)
        
        return self.curr_state()

    def _render(self, mode='human', close=False):
        return

    def seed(self, seed):
        random.seed(seed)
        np.random.seed
