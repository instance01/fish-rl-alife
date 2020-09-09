import numpy as np
from env.animal import Animal
import env.util as util
from abc import ABC
from env.animal_controller import RandomSharkController, DefaultSharkController


class Shark(Animal, ABC):

    COLOR = (255, 140, 0)  # orange
    RADIUS = 1.0
    FRICTION = 0.10  # high values decrease acceleration, maximum speed and turning circle
    MAX_SPEED_CHANGE = 0.06  # high values increase acceleration, maximum speed and turning circle
    MAX_ORIENTATION_CHANGE = float(np.radians(10.0))  # high values decrease the turning circle
    VIEW_DISTANCE = 100.0
    PROLONGED_SURVIVAL_PER_EATEN_FISH = 75
    INITIAL_SURVIVAL_TIME = 5000
    PROCREATE_AFTER_N_EATEN_FISH = 5
    CONTROLLER = None

    def __init__(self,
                 identifier: int,
                 position: np.array,
                 orientation: float,
                 ):

        Animal.__init__(self,
                        identifier,
                        position,
                        orientation,
                        self.RADIUS,
                        self.VIEW_DISTANCE,
                        self.FRICTION,
                        self.MAX_SPEED_CHANGE,
                        self.MAX_ORIENTATION_CHANGE,
                        self.COLOR,
                        )

        self.controller = self.CONTROLLER
        self.children = 0
        self.eaten_fish = 0

    # Override
    def is_ready_to_procreate(self) -> bool:
        return self.eaten_fish - (self.children * self.PROCREATE_AFTER_N_EATEN_FISH) > self.PROCREATE_AFTER_N_EATEN_FISH

    # Override
    def procreate(self, identifier: int) -> Animal:
        new_position = np.array([self.position[0] + 0.1 * self.radius, self.position[1]])
        start_speed, orientation = util.cartesian_to_polar(*self.velocity)
        new_shark = Shark(identifier, new_position, start_speed, orientation)
        self.children += 1
        return new_shark

    def is_starving(self):
        max_survival_time = self.INITIAL_SURVIVAL_TIME + self.eaten_fish * self.PROLONGED_SURVIVAL_PER_EATEN_FISH
        return (self.survived_n_steps / max_survival_time) >= 1

    # Override
    def name(self):
        return 'Shark.' + str(self.__identifier)

    def get_action(self, **observation):
        return self.controller.get_action(**observation)


class ControllerShark(Shark, ABC):

    CONTROLLER = None

    def __init__(self, *args, **kwargs):
        Shark.__init__(self, *args, **kwargs)
        self.controller = self.CONTROLLER

    def get_action(self, **observation):
        return self.controller.get_action(**observation)


class RandomShark(ControllerShark):

    CONTROLLER = RandomSharkController

    def name(self):
        return "RandomShark." + str(format(self.identifier))


class DefaultShark(ControllerShark):

    CONTROLLER = DefaultSharkController

    def name(self):
        return "DefaultShark." + str(format(self.identifier))


class SharkAgent(Shark):
    """Provided for Name Space"""
    pass





