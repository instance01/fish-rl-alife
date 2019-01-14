#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import logging
import sys
import time
from copy import deepcopy
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

import agentModules
from shark import Shark
from view import View


# distanz als observation
# absoluter anstatt relativer drehwinkel
# static learner nach gewisser anzahl episoden tauschen
# static learner nur tauschen, wenn besser?
# fische wuerfeln
# wand nicht bestaften --
# mehr fische
# hai kann nur kurz beschleunigen
# hai und fische trainen
# hai sieht nur radius
# hai bleibt laenger an dem zufallstaget
# fische sehen nur 5 nachbarfische
# fische sehen die Bewegung der anderen vor ihrem Zug oder nicht
# fishe müssen platz einnehmen, wenn sie kollidieren, setze ich beide zurück oder nur einen oder zufällig?
# fisch observations mischen
# fische sehen richtung und abstand anstatt absolute koordinaten

class FishDomainEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 2
    }
    
    def __init__(self, agentType, fish = 120, observableFish = 1, torus = True):
        self.width = 60
        self.height = self.width
        self.screen_width = 800
        self.screen_height = self.screen_width
        
        self.viewer = None
        self.close = False
        
        self.shark = 1
        self.fish = fish
        if observableFish == 0 or observableFish > fish:
            self.observableFish = fish
        else:
            self.observableFish = observableFish
        self.torus = torus
        self.learningAgent = None
        self.staticAgent = None
       
        self.sharks = []
        self.fishs = []
        
        
        self.agentType = agentType
        self.numActions = 5
#         self.joinedActions = self.numActions ** fish
        
#         actionSpace = [34] * fish
#         self.action_space = spaces.MultiDiscrete(actionSpace)
#         self.action_space = spaces.Discrete(self.joinedActions)
        self.action_space = agentModules.initActionSpace[self.agentType](self.numActions)
#         self.action_space = spaces.Box(low=-175.0, high=175.0, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observableFish + self.shark, 3))
        
        self.steps = 0
        self.episodes = 0
        
        self.caughtFish = 0
        self.caughtFishTimestamp = time.time()
    
    def setAgents(self, agent, staticAgent):
        self.learningAgent = agent
        self.staticAgent = staticAgent
        
            
    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.sharks = []
        for s in range(self.shark):
            self.sharks.append(Shark(self.rng, self.width, self.height, self.torus))
        self.fishs = []
        for f in range(self.fish):
            self.fishs.append(agentModules.createFish[self.agentType](self.rng, self.width, self.height))
        self.fishs[0].color = (0, 102, 0)    
            
        
        if self.staticAgent != None:
#             if self.episodes % 100 == 0:
            agentModules.cloneAgent[self.agentType](self.staticAgent, self.learningAgent)
        
        self.steps = 0
        self.episodes += 1

        self.evalCaughtFish()
        
        
        sharkPositions, sharkAngles, fishPositions, fishVelocities, fishAngles = self.getStats()
#         distances, directions = None, None
#         if self.observableFish != self.fish:

        distances, directions = self.getDistancesDirections(fishPositions, fishVelocities, sharkPositions)
        return self.getObservationDistancesDirections(0, sharkPositions, sharkAngles, fishPositions, fishAngles, distances, directions)
    
    
    def evalCaughtFish(self):
        minutes = (time.time() - self.caughtFishTimestamp) / 60.0
        print(self.caughtFish/ minutes)
        
        self.caughtFishTimestamp = time.time()
        self.caughtFish = 0
    
    
    def evalKernelDensity(self):
        def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
            """Build 2D kernel density estimate (KDE)."""
        
            # create grid of sample locations (default: 100x100)
            xx, yy = np.mgrid[x.min():x.max():xbins, 
                              y.min():y.max():ybins]
        
            xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
            xy_train  = np.vstack([y, x]).T
        
            kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
            kde_skl.fit(xy_train)
        
            # score_samples() returns the log-likelihood of the samples
            z = np.exp(kde_skl.score_samples(xy_sample))
            return xx, yy, np.reshape(z, xx.shape)
    
        sharkPositions, _, fishPositions, _, _ = self.getStats() 
        x,y = zip(*fishPositions)
        x,y = np.array(x)/float(self.width), np.array(y)/float(self.width)
        
        xx, yy, zz = kde2D(x, y, .05)

        plt.pcolormesh(xx, yy, zz)
        plt.scatter(x, y, s=2, facecolor='white')
        plt.scatter(sharkPositions[0][0]/float(self.width), sharkPositions[0][1]/float(self.height), s=2, facecolor='red')
        plt.show()
    
    
    def getStats(self):
        sharkPositions = [(s.x, s.y) for s in self.sharks]
        #sharkVelocities = [(s.velocity_x, s.velocity_y) for s in self.sharks]
        sharkAngles = [s.angle for s in self.sharks]
        fishPositions = [(f.x, f.y) for f in self.fishs]
        fishVelocities = [(f.velocity_x, f.velocity_y) for f in self.fishs]
        fishAngles = [f.angle for f in self.fishs]
        
        return sharkPositions, sharkAngles, fishPositions, fishVelocities, fishAngles
    
    def getObservationPositions(self, fishOwnPos = 0, sharkPositions = None, sharkAngles = None, fishPositions = None, fishAngles = None, distances = None):
#         print distances
        
        # make shallow copies of the arrays because the originals are used by all fish and we will shuffle and sort them
        # also the distances between the fish were calculated on the original order
        pos = sharkPositions[:]
        ang = sharkAngles[:]
        
        # FIXME this way the fishes see the sharks direclty after they moved which kind of breaks the mdp but it seems to help the learning
#         pos = [(s.x, s.y) for s in self.sharks]
#         ang = [[s.angle] for s in self.sharks]
            
            
        if self.observableFish != self.fish:
            # distances of a certain fish to all other fishes (except the shark (index 0) because it will get the sharks pos anyway)
            # those are sorted and we get the order of the indexes
            distArgs = np.argsort(distances[fishOwnPos])
            
#             Farbe der Fische zurücksetzen
#             if self.viewer and fishOwnPos == 0:
#                 for f in self.fishs[1:]:
#                     f.color = (154, 205, 50)
            
            # fish is closest to itself which is the first index in this sorted array
            for idx in distArgs[:self.observableFish]:
                pos.append(fishPositions[idx])
                ang.append(fishAngles[idx])

#                   die nächsten (sichtbaren) Fische zum ersten in der Liste markiern  
#                 if self.viewer and fishOwnPos == 0 and idx != 0:
#                     self.fishs[idx].color = (204, 51, 255)
            
#             print
#             print fishOwnPos, sharkPositions, fishPositions, distArgs, pos
            
            return np.hstack((np.array(pos) / self.width, np.array(ang) / 360.0)) 

        else:
            # the observation of the fish that is about to act should be at position after the sharks in the observations
            # that's why we swap
            fishPositions[0], fishPositions[fishOwnPos] = fishPositions[fishOwnPos], fishPositions[0]
            fishAngles[0], fishAngles[fishOwnPos] = fishAngles[fishOwnPos], fishAngles[0]
            
            return np.hstack((np.array(pos + fishPositions) / self.width, np.array(ang + fishAngles) / 360.0))   

    def getObservationDistancesDirections(self, fishOwnPos = 0, sharkPositions = None, sharkAngles = None, fishPositions = None, fishAngles = None, distances = None, directions = None):
        dist = distances[fishOwnPos, -self.shark:]
        dir = directions[fishOwnPos, -self.shark:]
        ang = sharkAngles[:]

        distArgs = np.argsort(distances[fishOwnPos, :-self.shark])
        
#             Farbe der Fische zurücksetzen
#             if self.viewer and fishOwnPos == 0:
#                 for f in self.fishs[1:]:
#                     f.color = (154, 205, 50)

        # fish is closest to itself which is the first index in this sorted array
        for idx in distArgs[:self.observableFish]:
            dist = np.hstack((dist, distances[fishOwnPos, idx]))
            dir = np.hstack((dir, directions[fishOwnPos, idx]))
            ang.append(fishAngles[idx])

#                   die nächsten (sichtbaren) Fische zum ersten in der Liste markiern  
#                 if self.viewer and fishOwnPos == 0 and idx != 0:
#                     self.fishs[idx].color = (204, 51, 255)

#             print(np.stack((dist, dir, np.array(ang))).transpose())
#             print(dist)
#             print(dir)
#             print(np.array(ang))

        # TODO If all fishes are observable, no iteration is needed as the right distances, directions can be selected
        
        return np.stack((dist / self.width, (dir + 180.0) / 360.0,  np.array(ang) / 360.0)).transpose()

            
    def getDistances(self, positions):
        if self.torus:
            distances = np.empty((0, self.fish))
            positions = np.array(positions)
            for fishPos in positions:
                absXY = np.absolute(positions - fishPos)
                absX = absXY[:,0]
                absY = absXY[:,1]
                
                diffAroundX = self.width - absX
                diffAroundY = self.height - absY
                
                # determine the nearest fish (also around the edges of the playing field
                dist = np.sqrt(np.square(np.minimum(absX, diffAroundX)) + np.square(np.minimum(absY, diffAroundY)))
                distances = np.append(distances, [dist], axis = 0)

            return distances
        else:
            positions = np.array(positions)
            b = positions.reshape(positions.shape[0], 1, positions.shape[1])
            return np.sqrt(np.einsum('ijk, ijk->ij', positions-b, positions-b))

    def getDistancesDirections(self, fishPositions, fishVelocities, sharkPositions):
        if self.torus:
            distances = np.empty((0, self.fish +  self.shark))
#             print(distances)
            directionAngles = np.empty((0, self.fish + self.shark))
            positions = np.array(fishPositions + sharkPositions)
            for idx, fishPos in enumerate(fishPositions):
                absXY = np.absolute(positions - fishPos)
                absX = absXY[:,0]
                absY = absXY[:,1]
                
                diffAroundX = self.width - absX
                diffAroundY = self.height - absY
                
                # determine the nearest fish (also around the edges of the playing field
                dist = np.sqrt(np.square(np.minimum(absX, diffAroundX)) + np.square(np.minimum(absY, diffAroundY)))
                distances = np.append(distances, [dist], axis = 0)

                # booleans for each fish if it is shorter to go direcly or over the edge in x- or y-direction 
                aroundX = (diffAroundX < absX)
                aroundY = (diffAroundY < absY)
                
                rightBelow = positions + np.transpose(np.array([self.width * aroundX, self.height * aroundY]))
                leftTop = positions - np.transpose(np.array([self.width * aroundX, self.height * aroundY]))
            
                dir = np.hstack((rightBelow, leftTop)).reshape(self.fish + self.shark, 2, 2)
                
                idxsOfSmallestDistance = np.argmin(np.linalg.norm(dir - fishPos, axis = 2), axis = 1)
                diffVectors = dir[range(len(idxsOfSmallestDistance)),idxsOfSmallestDistance,:] - fishPos
            
                cos = np.dot(fishVelocities[idx], np.transpose(diffVectors)) 
                sin = np.cross(fishVelocities[idx], np.transpose(diffVectors), axis = 0)
                atan2 = np.arctan2(sin, cos)
                dirAngles = np.degrees(atan2)
                
                directionAngles = np.append(directionAngles, [dirAngles], axis = 0)
                
            np.fill_diagonal(directionAngles, 0.0)
            
            return distances, directionAngles
        else:
            #FIXME: Qickfix with aroundX = [False,...] and aroundY = [False,...], could be calculated easier
            
            distances = np.empty((0, self.fish +  self.shark))
#             print(distances)
            directionAngles = np.empty((0, self.fish + self.shark))
            positions = np.array(fishPositions + sharkPositions)
            for idx, fishPos in enumerate(fishPositions):
                absXY = np.absolute(positions - fishPos)
                absX = absXY[:,0]
                absY = absXY[:,1]
                
                diffAroundX = self.width - absX
                diffAroundY = self.height - absY
                
                # determine the nearest fish (also around the edges of the playing field
                dist = np.sqrt(np.square(np.minimum(absX, diffAroundX)) + np.square(np.minimum(absY, diffAroundY)))
                distances = np.append(distances, [dist], axis = 0)

                # booleans for each fish if it is shorter to go direcly or over the edge in x- or y-direction 
                aroundX = np.full((self.fish +  self.shark), False, dtype=bool)
                aroundY = np.full((self.fish +  self.shark), False, dtype=bool)
                
                rightBelow = positions + np.transpose(np.array([self.width * aroundX, self.height * aroundY]))
                leftTop = positions - np.transpose(np.array([self.width * aroundX, self.height * aroundY]))
            
                dir = np.hstack((rightBelow, leftTop)).reshape(self.fish + self.shark, 2, 2)
                
                idxsOfSmallestDistance = np.argmin(np.linalg.norm(dir - fishPos, axis = 2), axis = 1)
                diffVectors = dir[range(len(idxsOfSmallestDistance)),idxsOfSmallestDistance,:] - fishPos
            
                cos = np.dot(fishVelocities[idx], np.transpose(diffVectors)) 
                sin = np.cross(fishVelocities[idx], np.transpose(diffVectors), axis = 0)
                atan2 = np.arctan2(sin, cos)
                dirAngles = np.degrees(atan2)
                
                directionAngles = np.append(directionAngles, [dirAngles], axis = 0)
                
            np.fill_diagonal(directionAngles, 0.0)
            
            return distances, directionAngles
    
    def step(self, action):
        
        # could also work with the last observation, that we gave to the learner
        sharkPositions, sharkAngles, fishPositions, fishVelocities, fishAngles = self.getStats()
#         distances, directions = None, None
#         if self.observableFish != self.fish:
        distances, directions = self.getDistancesDirections(fishPositions, fishVelocities, sharkPositions)
            
#         velocities = np.array([(a.velocity_x, a.velocity_y) for a in self.animals])
        
        # the joint action is converted into base_{self.numActions} which results in an action for each fish
        # actions = np.base_repr(actions, base=self.numActions).zfill(self.fish)
#         for idx, a in enumerate(actions):
#             self.animals[idx + 1].act(int(a))  # idx 0 is the shark
        for s in self.sharks:
            s.act(np.array(fishPositions))

        
        self.fishs[0].act(action, 0, distances, directions) # first fish is controlled by the learner
        
        # debug print
#         for idx, a in enumerate(self.fishs):
#             self.getObservationDistancesDirections(idx, sharkPositions, sharkAngles, fishPositions, fishAngles, distances, directions)
        
        for idx, a in enumerate(self.fishs[1:], start = 1):
            if self.staticAgent != None:
                obs = self.getObservationDistancesDirections(idx, sharkPositions, sharkAngles, fishPositions, fishAngles, distances, directions)
                choosenAction = self.staticAgent.forward(obs)
                a.act(choosenAction, idx, distances, directions)
            else:
                # if there's no second learning agent, all fish do the same action
                a.act(action, idx, distances, directions)
            
#         self.solveFishCollisions()
        
            
        reward = 1
        done = False
        
        sharkCollision, wallCollision = self.collisionDetect()
        
        if wallCollision:
            reward = -1
        
        if  sharkCollision:
            reward = -1000
            done = True
            
        if self.steps > 10000:
            done = True
        
        self.steps += 1 
        
        
        if self.steps % 100 == 0:
#             pass
            self.evalCaughtFish()
            self.evalKernelDensity()
        
        
#        print action, self.getObservationDistancesDirections(), reward
        sharkPositions, sharkAngles, fishPositions, fishVelocities, fishAngles = self.getStats()
#         if self.observableFish != self.fish:
        distances, directions = self.getDistancesDirections(fishPositions, fishVelocities, sharkPositions)
#         print(self.getObservationPositions(0, sharkPositions, sharkAngles, fishPositions, fishAngles, distances))
        return  self.getObservationDistancesDirections(0, sharkPositions, sharkAngles, fishPositions, fishAngles, distances, directions), reward, done, {}
                
    def render(self, mode='human', close=False):
        if close or self.close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        
        
        if self.viewer == None:
            scale = float(self.screen_width) / self.width
            self.viewer = View(self.screen_width, self.screen_height, scale, "FishDomain: " + self.agentType)  # self.__class__.__name__
         
        self.close = self.viewer.draw(self.sharks + self.fishs)
        
        return
            
    def collisionDetect(self):
        sharkCollision = False
        wallCollision = False
        
        for idx, a in enumerate(self.sharks + self.fishs):
#             if a.x < a.radius or a.x > self.width - a.radius:    a.velocity_x *= -1
#             if a.y < a.radius or a.y > self.height - a.radius:   a.velocity_y *= -1
            if (a.x < a.radius or
                a.x > self.width - a.radius or 
                a.y < a.radius or 
                a.y > self.height - a.radius): 
                if self.torus:
                    a.x %= self.width
                    a.y %= self.height
                else:
                    a.x -= a.velocity_x
                    a.y -= a.velocity_y
                    a.crashed = True
                    if idx == len(self.sharks) + 1:
                        wallCollision = True

        # FIXME Shark collisions are not correctly calculated around the edges in a torus (only collisions inside the field are considered)
        for s in self.sharks: 
            fishPositions = [(f.x, f.y) for f in self.fishs]
            sharkPos = np.array([s.x, s.y])
            distances = np.linalg.norm(fishPositions - sharkPos, axis=1)
            # get indexes of fish which are closer than 15 pixel
            # delete in reverse order (highest index first) because list gets shorter
            # add new fish?
            for index in sorted(np.argwhere(distances < 2), reverse=True):
                self.caughtFish += 1
                
                # if the first fish is hit, we keep it and end the episode
                if index[0] == 0:
                    sharkCollision = True
                else:
                    # else we delete and recreate the fish
                    del self.fishs[index[0]]
                    self.fishs.append(agentModules.createFish[self.agentType](self.rng, self.width, self.height))
                
                for ss in self.sharks:
                    # the fixed (for some steps) target of the shark is gone
                    if index[0] == ss.targetID:
                        ss.targetID = None
        
        return (sharkCollision, wallCollision)
    
    def solveFishCollisions(self):
        _, _, fishPositions, _, _ = self.getStats()
        fishDistances = self.getDistances(fishPositions)
        
        # iterate over lines and columns of the upper triangular matrix
        l, c = fishDistances.shape
        for i in range(l):
            for j in range(c):
                if i < j:
                    if fishDistances[i,j] < 2:
                        self.fishs[i].x -= self.fishs[i].velocity_x
                        self.fishs[i].y -= self.fishs[i].velocity_y
                        self.fishs[j].x -= self.fishs[j].velocity_x
                        self.fishs[j].y -= self.fishs[j].velocity_y
        

    def __getstate__(self):
        return
     
    def __setstate__(self, state):
        pass

