import math
import numpy as np

class Animal:
    def __init__(self, rng, width, height):
        self.rng = rng
        self.width = width
        self.height = height
        self.x = rng.randint(self.radius, width - self.radius)
        self.y = rng.randint(self.radius, height - self.radius)
        self.angle = rng.rand() * 360.0
        self.velocity_x, self.velocity_y = self.rotatedVector((self.velocity, 0.0), self.angle)
        self.crashed = False
        
    
    def rotatedVector(self, vector, degree):
        cos = math.cos(math.radians(degree))
        sin = math.sin(math.radians(degree))
        x = vector[0] * cos - vector[1] * sin
        y = vector[0] * sin + vector[1] * cos
        return (x, y)
    
    def getAngle(self, vector1, vector2):
        """Returns the angle between two vectors in range -180 ... 180 degree"""
        
#         vector1 = vector1/np.linalg.norm(vector1)
#         vector2 = vector2/np.linalg.norm(vector2)
        
        # https://newtonexcelbach.com/2014/03/01/the-angle-between-two-vectors-python-version/
        cos = np.dot(vector1, vector2)
        sin = np.cross(vector1, vector2)
         
        # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
#         cos = vector1[0] * vector2[0] + vector1[1] * vector2[1]
#         sin = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        
        atan2 = np.arctan2(sin, cos)
        degree = math.degrees(atan2)
        
        # http://wikicode.wikidot.com/get-angle-of-line-between-two-points
        # print math.degrees(np.arctan2(vector1[1], vector1[0]) - np.arctan2(vector2[1], vector2[0]))
        
        # could be done without angles and arctan by modifying the direction vector
        # https://gamedev.stackexchange.com/questions/7131/how-can-i-calculate-the-angle-and-proper-turn-direction-between-two-2d-vectors
        
#         print vector1, vector2, cos, sin, atan2, degree
        return degree
    
    
    def act(self, positions):
        raise NotImplementedError("Please implement \"act()\"")
    
    def move(self):
        self.x += self.velocity_x
        self.y += self.velocity_y

    def rotate(self, degree):
        self.angle += degree
        self.angle %= 360.0
        self.velocity_x, self.velocity_y = self.rotatedVector((self.velocity, 0.0), self.angle)
        
