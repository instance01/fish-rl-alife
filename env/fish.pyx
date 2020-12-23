#cython: language_level=3, boundscheck=False
import numpy as np
from env.animal import Animal
import env.util as util
from env.animal_controller import RandomFishController
from env.animal_controller import DemoFishController
from env.animal_controller import TurnAwayFishController
from env.animal_controller import BoidFishController
from copy import deepcopy
from abc import ABC


class Fish(Animal, ABC):
    COLOR = (51, 204, 59)  # Light green.
    RADIUS = 1.0
    # High values decrease acceleration, maximum speed and turning circle.
    FRICTION = 0.08
    # High values increase acceleration, maximum speed and turning circle.
    MAX_SPEED_CHANGE = 0.04
    # High values decrease the turning circle.
    MAX_ORIENTATION_CHANGE = float(np.radians(180.0))
    VIEW_DISTANCE = 10.0
    DRAW_VIEW_DISTANCE = False  # Drawn circle around animal.
    PROCREATE_AFTER_N_STEPS = 100

    def __init__(
        self,
        identifier: int,
        position: np.array,
        orientation: float,
    ):
        Animal.__init__(
            self,
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
        self.children = 0
        self.survived_steps = 0

    def name(self):
        return 'Fish.' + str(self._identifier)

    def is_ready_to_procreate(self) -> bool:
        return self.survived_n_steps - (self.children * self.PROCREATE_AFTER_N_STEPS) > self.PROCREATE_AFTER_N_STEPS

    def procreate(self, identifier: int) -> Animal:
        noise = ([0.1, 0.1], [0.1, 0.0], [0.0, 0.1])[np.random.randint(0, 3)]
        new_position = deepcopy(self.position) + noise
        start_speed, orientation = util.cartesian_to_polar(*self.velocity)
        # TODO: That was one of my fixes. Consider adding back.
        #new_fish = type(self)(identifier, new_position, start_speed, orientation)
        new_fish = type(self)(identifier, new_position, orientation)
        self.children += 1
        return new_fish


class ControllerFish(Fish, ABC):
    CONTROLLER = None

    def __init__(self, *args):
        Fish.__init__(self, *args)
        self.controller = self.CONTROLLER

    def get_action(self, **observation):
        return self.controller.get_action(**observation)


class RandomFish(ControllerFish):
    CONTROLLER = RandomFishController

    def name(self):
        return 'RandomFish.' + str(self._identifier)


class DemoFish(ControllerFish):
    CONTROLLER = DemoFishController

    def name(self):
        return 'DemoFish.' + str(self._identifier)


class TurnAwayFish(ControllerFish):
    CONTROLLER = TurnAwayFishController

    def name(self):
        return 'TurnAwayFish.' + str(self._identifier)


class BoidFish(ControllerFish):
    CONTROLLER = BoidFishController

    def name(self):
        return 'BoidFish.' + str(self._identifier)


class FishAgent(Fish):
    """Provided for Name Space"""
    pass
