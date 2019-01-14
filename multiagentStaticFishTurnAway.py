#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import copy
from fishDomain.envs.fishDomain import FishDomainEnv

class MultiagentStaticFishTurnAway():
    def __init__(self, width):
        self.width = width
    
    def forward(self, obs):
        if (obs[0, 0] * self.width) < (self.width/2.1):
            return ((obs[0, 1] * 360))
        else:
            return 0
        



env = FishDomainEnv("turnAway", 100, 1)
env.seed(2)

agent = MultiagentStaticFishTurnAway(env.width)
env.setAgents(agent, copy.deepcopy(agent))

 
for i_episode in range(1000):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = agent.forward(observation)
#         print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break