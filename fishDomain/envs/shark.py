from animal import Animal
import numpy as np

class Shark(Animal):
    def __init__(self, rng, width, height, torus):
        self.radius = 1
        self.velocity = 1.0
        self.color = (255, 140, 0)
        self.maxAngle = 45.0
        self.torus = torus
        self.steps = rng.randint(0, 100)
        self.targetID = None
        Animal.__init__(self, rng, width, height)
        
    def chooseTarget(self, distances, argmin):
        if self.steps % 20 == 0 or self.targetID == None:
            # get fish whose distance is < 1.5 of the distance of the closest fish
#             i = np.argwhere(distances <= distances[argmin] * 1.5)
            i = np.argwhere(distances <= 0.2 * self.width)
            
            if i.size == 0:
                i = np.array([argmin])
            
            # return random fish index
            choice = self.rng.choice(i.flatten()) 
#             choice = self.rng.choice(np.append(i.flatten(), [0])) # shark prefers fish 0

            self.targetID = choice
        
        return self.targetID
        
        
    def act(self, positions):
        # get nearest fish
        
        argmin = None
        diffVector = None
        ownPos = np.array([self.x, self.y])
        
        if not self.torus:
            distances = np.linalg.norm(positions - ownPos, axis=1)
            argmin = np.argmin(distances)
            argmin = self.chooseTarget(distances, argmin)
    #         print np.linalg.norm(positions[:-1] - positions[-1], axis=1), argmin, positions[argmin]
            diffVector = positions[argmin] - ownPos
        else:
            # sqrt(min(|x1 - x2|, w - |x1 - x2|)^2 + min(|y1 - y2|, h - |y1 - y2|)^2)
            # https://stackoverflow.com/questions/2123947/calculate-distance-between-two-x-y-coordinates
            absXY = np.absolute(positions - ownPos)
            absX = absXY[:,0]
            absY = absXY[:,1]
            
            diffAroundX = self.width - absX
            diffAroundY = self.height - absY
            
            # determine the nearest fish (also around the edges of the playing field
            distances = np.sqrt(np.square(np.minimum(absX, diffAroundX)) + np.square(np.minimum(absY, diffAroundY)))
            argmin = np.argmin(distances)
            argmin = self.chooseTarget(distances, argmin)
            
            # booleans for each fish if it is shorter to go direcly or over the edge in x- or y-direction 
            aroundX = (diffAroundX < absX)
            aroundY = (diffAroundY < absY)
            
#             print aroundX, aroundY
            # if it is shorter to go around the edge, pretend that the fish is also in those adjacent squares to calculate the direction
            rightBelow = positions[argmin] + [self.width * aroundX[argmin], self.height * aroundY[argmin]]
            leftTop = positions[argmin] - [self.width * aroundX[argmin], self.height * aroundY[argmin]]
            dir = np.array([rightBelow, leftTop])
            
            argmin2 = np.argmin(np.linalg.norm(dir - ownPos, axis=1))
            diffVector = dir[argmin2] - ownPos
            
        
        directionVector = (self.velocity_x, self.velocity_y)
        angle = self.getAngle(directionVector, diffVector)
        
        
        if abs(angle) > self.maxAngle:
            angle = angle/abs(angle) * self.maxAngle
        
        self.rotate(angle)
        
        if not self.torus and self.crashed and self.steps % 5 == 0:
            self.rotate(self.rng.uniform(-90.0, 91.0))
            self.crashed = False
        
        self.move()
        
        
    def move(self):
        self.steps += 1
        if self.steps % 20 == 0:
            self.velocity = 1.0
        if self.steps % 80 == 0:
            self.velocity = 1.6
        
        Animal.move(self)