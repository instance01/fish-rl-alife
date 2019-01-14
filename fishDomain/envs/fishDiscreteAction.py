from animal import Animal
import numpy as np

class FishDiscreteAction(Animal):
    def __init__(self, rng, width, height):
        self.radius = 1
        self.velocity = 1.0
        self.color = (154, 205, 50)
        Animal.__init__(self, rng, width, height)
        
    
    def act(self, action, idx,  distances, directions):
#         print action
#         if self.crashed:
#             self.rotate(self.rng.uniform(-100.0, 100.0))
#             self.crashed = False
#         else:
#             if distances[idx, -1] < 60.0:
#                 self.rotate(directions[idx, -1] + 180)
                
#             diffVector = positions[0] - (self.x, self.y)
#             if np.linalg.norm(diffVector) < 100:
#                 directionVector = (self.velocity_x, self.velocity_y)
#                 self.rotate(self.getAngle(directionVector, diffVector) + 180)
#                     self.rotate(self.getAngle(directionVector, velocities[-1]))


        # the joint action (rotation and velocity) is split which results in the two different actions
#         a = np.base_repr(action, base=5).zfill(2)
#         self.rotate(range(-90, 100, 45)[int(a[0])])
#         self.velocity = [x / 100.0 for x in range(0, 101, 25)][int(a[1])]

        self.rotate(range(-90, 100, 45)[action])
#         self.rotate(action)
        self.move()
