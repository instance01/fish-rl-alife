from fishDiscreteAction import FishDiscreteAction

class FishContinuousAction(FishDiscreteAction):
    def __init__(self, rng, width, height):
        FishDiscreteAction.__init__(self, rng, width, height)
        
    
    def act(self, action, idx,  distances, directions):
        self.rotate(action)
        self.move()
