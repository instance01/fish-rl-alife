from gym import spaces
from rl.util import *

from fishDomain import FishDomainEnv
from shark import Shark
from fishContinuousAction import FishContinuousAction

class FishDomainPolicyGradientEnv(FishDomainEnv):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 2
    }
    
    def __init__(self, agentType, fish = 3, observableFish = 5, torus = True):
        FishDomainEnv.__init__(self, agentType, fish, observableFish, torus)
        self.action_space = spaces.Box(low=-90.0, high=90.1, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observableFish + self.shark, 3))

    def reset(self):
        self.sharks = []
        for s in range(self.shark):
            self.sharks.append(Shark(self.rng, self.width, self.height, self.torus))
        self.fishs = []
        for f in range(self.fish):
            self.fishs.append(FishContinuousAction(self.rng, self.width, self.height))
            
        self.fishs[0].color = (0, 102, 0)   
        
        if self.staticAgent != None:
#             if self.episodes % 100 == 0:
            self.staticAgent.actor = clone_model(self.learningAgent.actor)
        
        self.steps = 0
        self.episodes += 1
        
        self.evalCaughtFish()
        
        sharkPositions, sharkAngles, fishPositions, fishVelocities, fishAngles = self.getStats()
#         distances, directions = None, None
#         if self.observableFish != self.fish:

        distances, directions = self.getDistancesDirections(fishPositions, fishVelocities, sharkPositions)
        return self.getObservationDistancesDirections(0, sharkPositions, sharkAngles, fishPositions, fishAngles, distances, directions)


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
                    self.fishs.append(FishContinuousAction(self.rng, self.width, self.height))
                
                for ss in self.sharks:
                    # the fixed (for some steps) target of the shark is gone
                    if index[0] == ss.targetID:
                        ss.targetID = None
        
        return (sharkCollision, wallCollision)