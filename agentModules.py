#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gym import spaces
from rl.util import *

from fishDomain.envs.fishDiscreteAction import FishDiscreteAction
from fishDomain.envs.fishContinuousAction import FishContinuousAction

def initDiscreteAction(numberActions):
    return spaces.Discrete(numberActions)
        
def initContinuousAction(_):
    return spaces.Box(low=-90.0, high=90.1, shape=(1,))






def cloneDqn(staticAgent, learningAgent):
    staticAgent.model.set_weights(learningAgent.model.get_weights())

def cloneDdpg(staticAgent, learningAgent):
    staticAgent.actor = clone_model(learningAgent.actor)

def cloneStaticFishTurn(staticAgent, learningAgent):
    pass






def fishDiscreteAction(rng, width, height):
    return FishDiscreteAction(rng, width, height)


def fishContinuousAction(rng, width, height):
    return FishContinuousAction(rng, width, height)






initActionSpace = {"dqn": initDiscreteAction, "ddpg": initContinuousAction, "turnAway": initContinuousAction}
cloneAgent = {"dqn": cloneDqn, "ddpg": cloneDdpg, "turnAway": cloneStaticFishTurn}
createFish = {"dqn": fishDiscreteAction, "ddpg": fishContinuousAction, "turnAway": fishContinuousAction}