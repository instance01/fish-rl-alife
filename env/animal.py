import numpy as np
import env.util as util
from abc import ABC, abstractmethod


class Animal(ABC):
    def __init__(
        self,
        identifier: int,
        position: np.array,
        orientation: float,
        radius: float,
        view_distance: float,
        friction: float,
        max_speed_change: float,
        max_orientation_change: float,
        color: (int, int, int)
    ):
        self._identifier = identifier
        self.position = position

        self.radius = radius
        self.view_distance = view_distance
        self.friction = friction

        self.max_speed_change = max_speed_change
        self.max_orientation_change = max_orientation_change

        self.velocity = np.asarray(
            util.polar_to_cartesian(0, orientation),
            "float32"
        )
        self.orientation = orientation

        self.color = color

        self.__speed_change = 0
        self.__orientation_change = 0

        self.survived_n_steps = 0

        self.stun_steps = 0

    @property
    def max_speed(self):
        return self.max_speed_change / self.friction

    @property
    def identifier(self):
        return self.identifier

    @property
    def __orientation(self):
        return np.arctan2(self.velocity[1], self.velocity[0])

    def move(self) -> None:
        # Update orientation.
        self.orientation = util.add_angles(
            self.orientation,
            self.__orientation_change
        )

        # Change velocity.
        movement_x, movement_y = util.polar_to_cartesian(
            self.__speed_change,
            self.orientation
        )
        self.velocity[0] += movement_x
        self.velocity[1] += movement_y

        # Apply water friction and update position.
        self.velocity *= (1 - self.friction)
        self.position += self.velocity

    def act(self, speed_change: float, orientation_change: float) -> None:
        # 1 -> max_speed
        # 1 -> -np.pi / np.pi
        self.__speed_change = np.clip(
            speed_change,
            -self.max_speed_change,
            self.max_speed_change
        )
        self.__orientation_change = np.clip(
            orientation_change,
            -self.max_orientation_change,
            self.max_orientation_change
        )
        self.move()

    # @abstractmethod
    def is_ready_to_procreate(self) -> bool:
        """
        States whether the animal is ready to procreate
        :return: True, if the animal can procreate
        """
        pass

    # @abstractmethod
    def procreate(self, identifier: int) -> object:
        """
        Creates a new Animal and returns it
        :return: a new Animal
        """
        pass

    # @abstractmethod
    def name(self):
        """
        returns a unique name for each animal
        :return: the name of the animal as string
        """
        pass

    def __hash__(self):
        return hash(self._identifier)

    def __eq__(self, other):
        if isinstance(other, Animal):
            return other._identifier == self._identifier \
                   and np.array_equal(other.position, self.position) \
                   and np.array_equal(other.velocity, self.velocity)
        else:
            return False
