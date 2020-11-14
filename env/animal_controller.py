import random
import numpy as np
import env.util as util
from abc import ABC, abstractmethod

random_state = np.random.RandomState()


class Controller(ABC):

    @staticmethod
    @abstractmethod
    def get_action(own_orientation: float, ready_to_procreate: float,
                   wall_observation: np.ndarray, shark_observation: np.ndarray,
                   fish_observation: np.ndarray) -> (float, float, float):
        pass


class RandomFishController(Controller):

    @staticmethod
    def get_action(**kargs) -> (float, float, float):
        speed_change = random_state.uniform(0, 1)
        direction_change = random_state.uniform(-0.5, 0.5)
        procreate = random_state.randint(0, 2)
        return speed_change, direction_change, procreate


class DemoFishController(Controller):

    @staticmethod
    def get_action(**kargs) -> (float, float, float):
        speed_change = 1.0
        direction_change = 0.5
        procreate = False
        return speed_change, direction_change, procreate


class BoidFishController(Controller):

    @staticmethod
    def get_action(own_orientation: float, ready_to_procreate: float, wall_observation: np.ndarray,
                   shark_observation: np.ndarray, fish_observation: np.ndarray) -> (float, float, float):

        distance_to_closest_wall, angle_to_closest_wall = wall_observation[0], wall_observation[1]
        distance_to_second_closest_wall, angle_to_second_closest_wall = wall_observation[2], wall_observation[3]

        distance_to_shark, angle_to_shark = shark_observation[6], shark_observation[7]
        angle_to_shark = util.scale(angle_to_shark, -1, 1, -np.pi, np.pi)
        shark_force = calculate_repel_force(distance_to_shark, angle_to_shark)

        # re-scale angles from [-1; 1] to [-pi; pi] to ease angle calculations
        own_orientation = util.scale(own_orientation, -1, 1, -np.pi, np.pi)
        angle_to_closest_wall = util.scale(angle_to_closest_wall, -1, 1, -np.pi, np.pi)
        angle_to_second_closest_wall = util.scale(angle_to_second_closest_wall, -1, 1, -np.pi, np.pi)

        # prepare boid parameters (alignment, separation, average_position)
        orientation_count = 0
        orientation_sum = np.array([0.0, 0.0])
        repulse_forces = np.array([0.0, 0.0])
        average_position = np.array([0.0, 0.0])
        average_position_count = 0

        # iterate to get neighbor data
        for i in range(0, len(fish_observation), 3):
            distance_to_fish = fish_observation[i]
            if distance_to_fish != 0:
                vector_to_fish = util.polar_to_cartesian(1, util.scale(fish_observation[i + 1], -1, 1, -np.pi, np.pi))
                fish_orientation_vector = util.polar_to_cartesian(1, util.scale(fish_observation[i + 2], -1, 1, -np.pi,
                                                                                np.pi))

                orientation_sum += fish_orientation_vector
                orientation_count += 1
                average_position += vector_to_fish
                average_position_count += 1
                repulse_forces -= vector_to_fish / distance_to_fish

        # boid forces
        alignment_force = np.array([0.0, 0.0])
        if orientation_count > 0:
            alignment_force = orientation_sum / orientation_count
        cohesion_force = np.array([0.0, 0.0])
        if average_position_count > 0:
            cohesion_force = average_position / average_position_count
        separation_force = repulse_forces

        # other factors
        wall1_force = calculate_repel_force(distance_to_closest_wall, angle_to_closest_wall)
        wall2_force = calculate_repel_force(distance_to_second_closest_wall, angle_to_second_closest_wall)
        noise_force = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])

        # weighted sum
        target_vector = 0.03 * separation_force + 0.2 * alignment_force + 0.6 * cohesion_force + 0.4 * shark_force + 1.0 * wall1_force + 1.0 * wall2_force

        angle = util.angle_of(target_vector)
        angle_change = util.scale(util.shortest_angle_between(own_orientation, angle), -np.pi, np.pi, -1.0, 1.0)
        speed_change = np.clip(util.length_of(target_vector), -1.0, 1.0)

        return speed_change, angle_change, ready_to_procreate and not distance_to_shark


class TurnAwayFishController(Controller):

    @staticmethod
    def get_action(own_orientation: float, ready_to_procreate: float,
                   wall_observation: np.ndarray, shark_observation: np.ndarray,
                   fish_observation: np.ndarray) -> (float, float, float):

        distance_to_closest_wall, angle_to_closest_wall = wall_observation[0], wall_observation[1]
        distance_to_second_closest_wall, angle_to_second_closest_wall = wall_observation[2], wall_observation[3]

        distance_to_shark, angle_to_shark = shark_observation[0], shark_observation[1]

        distance_to_closest_fish = fish_observation[0]
        angle_to_closest_fish = fish_observation[1]

        # re-scale angles from [-1; 1] to [-pi; pi] to ease angle calculations
        own_orientation = util.scale(own_orientation, -1, 1, -np.pi, np.pi)
        angle_to_closest_wall = util.scale(angle_to_closest_wall, -1, 1, -np.pi, np.pi)
        angle_to_second_closest_wall = util.scale(angle_to_second_closest_wall, -1, 1, -np.pi, np.pi)
        angle_to_shark = util.scale(angle_to_shark, -1, 1, -np.pi, np.pi)
        angle_to_closest_fish = util.scale(angle_to_closest_fish, -1, 1, -np.pi, np.pi)

        # other factors
        current_vector = np.array(util.polar_to_cartesian(1.0, own_orientation))
        fish_force = calculate_repel_force(distance_to_closest_fish, angle_to_closest_fish)
        shark_force = calculate_repel_force(distance_to_shark, angle_to_shark)
        wall1_force = calculate_repel_force(distance_to_closest_wall, angle_to_closest_wall)
        wall2_force = calculate_repel_force(distance_to_second_closest_wall, angle_to_second_closest_wall)
        noise_force = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        
        #weighted sum
        target_vector = current_vector + shark_force + 0.5 * fish_force + 0.5 * wall1_force + 0.3 * wall2_force
        angle = util.angle_of(target_vector)
        if angle:
            angle_change = util.scale(util.shortest_angle_between(own_orientation, angle), -np.pi, np.pi, -1.0, 1.0)
        else:
            angle_change = 0.0

        return 1.0, angle_change, ready_to_procreate == 1.0 and not distance_to_shark


class RandomSharkController(Controller):

    @staticmethod
    def get_action(**kargs) -> (float, float, float):
        speed_change = random_state.uniform(0, 1)
        direction_change = random_state.uniform(-0.5, 0.5)
        procreate = random_state.randint(0, 2)
        return speed_change, direction_change, procreate


class DefaultSharkController(Controller):

    @staticmethod
    def get_action(own_orientation: float, ready_to_procreate: float, wall_observation: np.ndarray,
                   shark_observation: np.ndarray, fish_observation: np.ndarray) -> (float, float, float):
        distance_to_closest_wall, angle_to_closest_wall = wall_observation[0], wall_observation[1]
        distance_to_second_closest_wall, angle_to_second_closest_wall = wall_observation[2], wall_observation[3]

        distance_to_closest_shark, angle_to_closest_shark = shark_observation[0], shark_observation[1]

        distance_to_fish, angle_to_fish = fish_observation[0], fish_observation[1]

        own_orientation = util.scale(own_orientation, -1, 1, -np.pi, np.pi)
        angle_to_closest_wall = util.scale(angle_to_closest_wall, -1, 1, -np.pi, np.pi)
        angle_to_second_closest_wall = util.scale(angle_to_second_closest_wall, -1, 1, -np.pi, np.pi)
        angle_to_fish = util.scale(angle_to_fish, -1, 1, -np.pi, np.pi)
        angle_to_closest_shark = util.scale(angle_to_closest_shark, -1, 1, -np.pi, np.pi)

        fish_force = -calculate_repel_force(distance_to_fish, angle_to_fish)
        wall1_force = calculate_repel_force(distance_to_closest_wall, angle_to_closest_wall)
        wall2_force = calculate_repel_force(distance_to_second_closest_wall, angle_to_second_closest_wall)
        shark_force = calculate_repel_force(distance_to_closest_shark, angle_to_closest_shark)

        target_vector = fish_force
        # if 0 < distance_to_closest_shark < 0.05:
        target_vector += 0.1 * shark_force

        # if 0 < distance_to_closest_wall < 0.01:
        target_vector += 0.0005 * (wall1_force + wall2_force)

        angle = util.angle_of(target_vector)
        angle_change = util.scale(util.shortest_angle_between(own_orientation, angle), -np.pi, np.pi, -1.0, 1.0)
        return 1.0, angle_change, False


# calculate wall repulsion force
def calculate_repel_force(distance: float, angle: float) -> np.array:
    if distance != 0:
        dx, dy = util.polar_to_cartesian(1, angle)
        return np.array([-dx / distance, -dy / distance])
    return np.array([0.0, 0.0])

