import inspect
import numpy as np
import bisect
import itertools as it
from copy import deepcopy

from env.animal import Animal
from env.fish import Fish
from env.shark import Shark
from env.view import View
import env.util as util
from env.collision import CollisionSpace
from env.fish import RandomFish, TurnAwayFish, BoidFish
from env.shark import RandomShark, DefaultShark, SharkAgent


WINDOW_NAME = "Aquarium"
FPS = 25
OBSERVATION_MIN = -1.0
OBSERVATION_MAX = 1.0
ACTION_MIN = -1.0
ACTION_MAX = 1.0


class Aquarium:
    def __init__(
        self,
        size: int = 40,
        observable_sharks: int = 2,
        observable_fishes: int = 3,
        observable_walls: int = 2,
        max_steps: int = 500,
        max_fish: int = 30,
        max_sharks: int = 5,
        water_friction: float = 0.08,
        torus: bool = False,
        fish_collision: bool = True,
        lock_screen: bool = False,
        seed=42,
        # TODO That was one of my fixes. Unused right now.
        nr_sharks=1,  # TODO unused
        nr_fishes=1  # TODO unused
    ):
        self.seed: int = seed
        np.random.seed(seed=seed)

        self.fishes: [Fish] = set()
        self.sharks: [Shark] = set()
        self.next_fish_id: int = 0
        self.next_shark_id: int = 0

        # Environment parameters.
        self.size: int = size
        self.height = self.width = size
        self.torus: bool = torus
        self.fish_collision: bool = fish_collision
        self.collision_space = CollisionSpace(
            self.torus,
            0,
            0,
            self.width,
            self.height
        )
        self.water_friction = water_friction

        # Observation and action space.
        self.observable_walls: int = observable_walls
        self.observable_fishes: int = observable_fishes
        self.observable_sharks: int = observable_sharks
        # Diagonal line | only valid for width = height.
        self.max_animal_view_distance: float = np.sqrt(2) * self.width

        # Distance, angle to wall.
        self.observations_per_wall: int = 2
        # Distance, angle to animal and orientation of animal.
        self.observations_per_animal: int = 3

        # TODO Why square of observable_walls? Shouldn't it be
        # observations_per_wall?
        # own_orientation, ready_to_procreate
        self.observation_length: int = 2 \
            + (self.observable_walls * self.observable_walls) \
            + (self.observable_fishes + self.observable_sharks) \
            * self.observations_per_animal

        # Environment limits.
        self.max_fish: int = max_fish
        self.max_sharks: int = max_sharks
        self.max_steps: int = max_steps

        # Dict mapping a shark to a reward(float).
        self.track_shark_reward = {Shark: float}
        self.track_fish_reward = {Fish: float}

        # Type of fish :: amount
        self.fish_types = {
            'RandomFish': 0,
            'TurnAwayFish': 0,
            'BoidFish': 0
        }

        # Type of shark :: amount
        self.shark_types = {
            'RandomShark': 0,
            'DefaultShark': 0
        }

        self.current_step: int = 0

        # Count the dead.
        self.dead_fishes: int = 0
        self.dead_sharks: int = 0

        # GUI
        self.close = lock_screen
        self.screen_height = self.screen_width = 800
        self.view = View(
            self.screen_width,
            self.screen_height,
            self.screen_width / self.width,
            WINDOW_NAME,
            FPS
        )

    @property
    def current_shark_population(self) -> int:
        return len(self.sharks)

    @property
    def current_fish_population(self) -> int:
        return len(self.fishes)

    @property
    def killed_fishes(self) -> int:
        return self.dead_fishes

    @property
    def killed_sharks(self) -> int:
        return self.dead_sharks

    def select_fish_types(self, Random_Fish=0, Turn_Away_Fish=0, Boid_Fish=0):
        self.fish_types['RandomFish'] = Random_Fish
        self.fish_types['TurnAwayFish'] = Turn_Away_Fish
        self.fish_types['BoidFish'] = Boid_Fish
        return self

    def select_shark_types(self, Shark_Agents=0):
        # TODO: Update the docstring below.
        """Derzeit gibt es not keine Möglichkeit, statische und normale Agent zu kombinieren
           Das kommt noch"""
        #self.shark_types['RandomShark'] = Random_Shark
        #self.shark_types['DefaultShark'] = Default_Shark
        self.shark_types['Shark_Agents'] = Shark_Agents
        return self

    def reset(self) -> np.array:
        self.current_step = 0

        self.next_shark_id = 0
        self.next_fish_id = 0

        self.sharks = set()
        self.fishes = set()

        self.dead_sharks = 0
        self.dead_fishes = 0

        # Initialize fishes at random positions.
        for f_type, amount in self.fish_types.items():
            fish = [self.create_fish(f_type) for _ in range(amount)]
            self.fishes.update(set(fish))

        # Initialize sharks at random positions.
        for s_type, amount in self.shark_types.items():
            shark = [self.create_shark(s_type) for _ in range(amount)]
            self.sharks.update(set(shark))

        if self.fishes:
            for f1, f2 in it.combinations(self.fishes, 2):
                if self.collision_space.check_collision(f1, f2):
                    self.collision_space.perform_collision(f1, f2)

        if self.sharks:
            for s1, s2 in it.combinations(self.sharks, 2):
                if self.collision_space.check_collision(s1, s2):
                    self.collision_space.perform_collision(s1, s2)

        return self.create_named_shark_observation()

    def create_fish(self, f_type: str):
        self.next_fish_id += 1

        # TODO Make this a global variable.
        types = {
            "RandomFish": RandomFish,
            "TurnAwayFish": TurnAwayFish,
            "BoidFish": BoidFish
        }

        x_pos = np.random.uniform(0, self.width)
        y_pos = np.random.uniform(0, self.height)
        position = np.array((x_pos, y_pos))
        orientation = np.random.uniform(-np.pi, np.pi)

        return types[f_type](self.next_fish_id, position, orientation)

    def create_shark(self, s_type: str):
        self.next_shark_id += 1

        # TODO Make this a global variable.
        types = {
            "RandomShark": RandomShark,
            "DefaultShark": DefaultShark,
            "Shark_Agents": SharkAgent
        }

        x_pos = np.random.uniform(0, self.width)
        y_pos = np.random.uniform(0, self.height)
        position = np.array((x_pos, y_pos))
        orientation = np.random.uniform(-np.pi, np.pi)

        return types[s_type](self.next_shark_id, position, orientation)

    def prepare_observation_for_controller(self, observation):
        w1 = 2
        w2 = 2 + self.observable_walls * self.observations_per_wall
        s2 = w2 + self.observations_per_animal * self.observable_sharks
        return {
            "own_orientation": observation[0],
            "ready_to_procreate": observation[1],
            "wall_observation": observation[w1:w2],
            "shark_observation": observation[w2:s2],
            "fish_observation": observation[s2:-1]
        }

    def step(self, joint_shark_action: {Fish: tuple}):
        msg = "joint_shark_action length {} != number of fishes {}"
        assert len(self.sharks) == len(joint_shark_action), msg.format(
            len(joint_shark_action), len(self.sharks)
        )

        self.track_shark_reward = {shark: 0 for shark in self.sharks}
        self.track_fish_reward = {fish: 0 for fish in self.fishes}

        # Update environment metrics.
        self.current_step += 1

        self.move_fishes()
        self.move_sharks(joint_shark_action)

        # GUI
        if not self.close:
            self.close = self.view.check_for_interrupt()
            if self.close:
                self.view.close()

        # Turn fish to name.
        shark_reward = {
            shark.name: reward
            for shark, reward in self.track_shark_reward.items()
        }

        # Dict of all killed or alive animals: True = done = dead.
        shark_done = {
            shark.name: not(shark in self.sharks)
            for shark in self.track_shark_reward.keys()
        }

        return self.create_named_shark_observation(), shark_reward, shark_done

    def move_fishes(self):
        """Move all fishes."""
        named_joint_fish_observation = self.create_named_fish_observation()

        new_fishes = []
        for fish in self.fishes:
            # Fish action.
            observation: np.ndarray = named_joint_fish_observation[fish.name]
            observation = self.prepare_observation_for_controller(observation)
            action = fish.get_action(**observation)

            speed: float = action[0]
            angle: float = action[1]
            procreate: bool = action[2]

            # Ignore movement if fish wants to procreate.
            if procreate:
                blocked = self.current_fish_population + len(new_fishes) > self.max_fish
                if not blocked and fish.is_ready_to_procreate():
                    # self.current_fish_population + len(new_fishes) < self.max_fish and fish.is_ready_to_procreate()
                    new_fish = fish.procreate(identifier=self.next_fish_id)
                    new_fish.parent = fish
                    new_fishes.append(new_fish)
                    self.next_fish_id += 1
            else:
                # Move current fish according to action values.
                speed = util.scale(speed, 0.0, 1.0, 0.0, fish.max_speed)
                angle = util.scale(angle, -1.0, 1.0, -np.pi, np.pi)
                fish.act(speed, angle)
                self.collision_space.perform_boundary_collision(fish)

        # Add new fish.
        self.fishes.update(set(new_fishes))

        if self.fish_collision and self.fishes:
            combinations = it.combinations(self.fishes, 2)
            for a1, a2 in combinations:
                if self.collision_space.check_collision(a1, a2):
                    self.collision_space.perform_collision(a1, a2)

    def move_sharks(self, joint_shark_action):
        """Move all sharks according to joint_shark_action.

        :param joint_shark_action: {shark_name: str, action: (int, int, int)}
               where
                    key:    shark_name = names of all shark to move
                    value:  action = (speed, angle, procreate)

        :type {str, tuple}
        """
        new_sharks = []
        starved_sharks = []
        for shark in self.sharks:
            action: tuple = joint_shark_action[shark.name]
            speed: float = action[0]
            angle: float = action[1]
            procreate: bool = action[2]

            # Don't remove shark while iterating over self.sharks! Do it later.
            if shark.is_starving():
                starved_sharks.append(shark)
            elif procreate:
                # Ignore movement if shark wants to procreate.
                blocked = self.current_shark_population + len(new_sharks) >= self.max_sharks
                if not blocked and shark.is_ready_to_procreate():
                    new_shark = shark.procreate(id=self.next_shark_id)
                    new_sharks.append(new_shark)
                    self.next_shark_id += 1
            else:
                # Move current shark according to action values.
                speed = util.scale(speed, -1.0, 1.0, -shark.max_speed, shark.max_speed)
                angle = util.scale(angle, -1.0, 1.0, -np.pi, np.pi)
                shark.act(speed, angle)
                shark.survived_n_steps += 1

                # Solve shark-wall-collisions.
                self.collision_space.perform_boundary_collision(shark)

        # Add new sharks.
        self.sharks.update(set(new_sharks))

        # Remove dead sharks.
        self.dead_sharks += len([self.sharks.remove(shark) for shark in starved_sharks])

        # Check for killed fish.
        if self.sharks and self.fishes:
            combinations = it.product(self.sharks, self.fishes)
            for shark, fish in combinations:
                if self.collision_space.check_collision(shark, fish):
                    self.fishes.remove(fish)
                    if shark in self.track_shark_reward:
                        self.track_shark_reward[shark] += 10
                else:
                    fish.survived_n_steps += 1

        # Solve shark-shark-collisions.
        if self.fish_collision and self.sharks:
            combinations = it.combinations(self.sharks, 2)
            for a1, a2 in combinations:
                if self.collision_space.check_collision(a1, a2):
                    self.collision_space.perform_collision(a1, a2)

    def create_named_shark_observation(self) -> np.array:
        """Creates a dict map the name of each shark to his corresponding
        observation.

        named_shark_observation = {
            "shark_1: [0.873421, 0.232241, ... , 0,132419, -0.346561],
            "shark_n": [ ... ]
        }

        :return: a dict {str: np.ndarray} shark_name: observation
        """
        # Prepare joint observation.
        named_observation = {}
        # Insert observation of (remaining !) sharks.
        for shark in self.sharks:
            named_observation[shark.name] = self.observe_environment(
                shark,
                not self.torus,
                True
            )
        return named_observation

    def create_named_fish_observation(self) -> np.array:
        """Creates a dict to map the name of each fish to his corresponding
        observation.
        This is a joint observation.

        named_fish_observation = {
            "fish_1: [0.873421, 0.232241, ... , 0,132419, -0.346561],
            "fish_n": [ ... ]
        }

        :return: a dict {str: np.ndarray} fish_name: observation
        """
        named_observation = {}
        # Insert observation of (remaining, i.e. not dead) fishes.
        for fish in self.fishes:
            named_observation[fish.name] = self.observe_environment(
                fish, not self.torus, True
            )
        return named_observation

    def build_animal_observation_list(
        self,
        observer,
        observe_animals,
        observable_animals,
        curr_population,
        animals
    ):
        """Build a list of animal observations.
        This includes automatic sorting based on distance and zero padding.
        An animal could for instance be a shark or a fish.
        """
        # Append animals to observation with the closest shark upfront.
        curr_observable_animals = []
        distant_animals = []
        for animal in animals:
            if animal is not observer and observe_animals:
                # Observation contains: distance, angle, orientation
                observation = self.observe_animal(observer, animal)
                # If a shark is outside of the observation range, the empty
                # entry is kept to preserve the offset for the following
                # observations.
            else:
                # Fish cant see themself.
                observation = [0] * self.observations_per_animal

            if observation[0] != 0:
                bisect.insort(curr_observable_animals, observation)
            else:
                distant_animals += observation
        observed_animals = list(it.chain(*curr_observable_animals)) + distant_animals

        # Cut observed animals to the length passed.
        observed_animals = observed_animals[
            :(self.observations_per_animal * observable_animals)
        ]

        # If fewer animals were observed than are observable, we need to zero pad.
        zero_padding = [0] * max(
            (observable_animals - curr_population) * self.observations_per_animal,
            0
        )
        return np.concatenate((observed_animals, zero_padding))

    def observe_environment(
        self,
        observer: Animal,
        observe_walls: bool,
        observe_animals: bool
    ) -> [float]:
        """Returns an np.ndarray representing the observation of the given
        Animal.

        :param observer: an Animal of which the observation shall be calculated
        :type Animal
        :param observe_walls: True, if wall shall be observed
        :type bool
        :param observe_animals: True, if other animals shall be observed
        :type bool
        :return: np.ndarray observation
        """
        # Append the observers' view and speed to the observation.
        observer_orientation = util.scale(
            observer.orientation,
            -np.pi,
            np.pi,
            OBSERVATION_MIN,
            OBSERVATION_MAX
        )
        observer_read_to_procreate = observer.is_ready_to_procreate()
        observer_data = [observer_orientation, observer_read_to_procreate]

        # Append the closest aquarium borders to observation with the closest
        # border in front of the observation;
        # if the borders are too far away, empty entries preserve the offset
        # for the following observations.
        if observe_walls:
            observed_borders = self.observe_borders(observer)
        else:
            observed_borders = [0.0] * (
                self.observable_walls * self.observations_per_wall
            )

        observed_sharks = self.build_animal_observation_list(
            observer,
            observe_animals,
            self.observable_sharks,  # Predefined number of observable sharks. TODO: Rename to n_observable_sharks?
            self.current_shark_population,
            self.sharks
        )

        observed_fishes = self.build_animal_observation_list(
            observer,
            observe_animals,
            self.observable_fishes,  # Predefined number of observable fishes. TODO: Rename to n_observable_fishes?
            self.current_fish_population,
            self.fishes
        )

        return np.concatenate(
            (
                observer_data,
                observed_borders,
                observed_sharks,
                observed_fishes,
            )
        )

    def observe_borders(self, observer: Animal) -> [float]:
        distance_to_right_wall = self.width - observer.position[0]
        angle_to_right_wall = 0

        distance_to_left_wall = observer.position[0]
        angle_to_left_wall = np.pi

        distance_to_bottom_wall = self.height - observer.position[1]
        angle_to_bottom_wall = 0.5 * np.pi

        distance_to_upper_wall = observer.position[1]
        angle_to_upper_wall = -0.5 * np.pi

        polar_coordinates_of_walls = [
            (distance_to_right_wall, angle_to_right_wall),
            (distance_to_left_wall, angle_to_left_wall),
            (distance_to_bottom_wall, angle_to_bottom_wall),
            (distance_to_upper_wall, angle_to_upper_wall)
        ]

        # Sort coordinates by distance to wall.
        polar_coordinates_of_walls = sorted(
            polar_coordinates_of_walls,
            key=lambda polar_coord: polar_coord[0]
        )

        # Filter for visible walls.
        visible_walls = [
            polar_coord
            for polar_coord in polar_coordinates_of_walls
            if polar_coord[0] <= observer.view_distance
        ]

        if self.observable_walls >= 2:
            max_view_distance = min(observer.view_distance, self.width / 2)
        else:
            max_view_distance = min(observer.view_distance, self.width)

        # Flattening the polar-coordinates.
        observation = []
        for (distance, angle) in visible_walls[:self.observable_walls]:
            observation.append(util.scale(distance, observer.radius, max_view_distance, 0, OBSERVATION_MAX))
            observation.append(util.scale(angle, -np.pi, np.pi, OBSERVATION_MIN, OBSERVATION_MAX))

        # Fill with zeros.
        observation += [0] * (self.observations_per_wall * self.observable_walls - len(observation))
        return observation

    def observe_animal(self, observer: Animal, animal: Animal) -> [float]:
        """Build an animal observation.
        This includes three components:
        * distance between observer and animal.
        * angle / direction from observer to the animal.
        * orientation of the animal.
        If the animal is out of range (too far away), the observation is a set
        of zeroes.
        """
        observation = [0.0] * self.observations_per_animal

        if self.torus:
            dx, dy = util.vector_in_torus_space(
                observer.position,
                animal.position,
                self.width,
                self.height
            )
        else:
            dx, dy = animal.position - observer.position
        distance, direction = util.cartesian_to_polar(dx, dy)

        if 0 < distance < observer.view_distance:
            if self.fish_collision:
                min_distance = observer.radius + animal.radius
            else:
                min_distance = 0
            max_distance = min(observer.view_distance, self.max_animal_view_distance)
            observation[0] = util.scale(distance, min_distance, max_distance, 0, OBSERVATION_MAX)
            observation[1] = util.scale(direction, -np.pi, np.pi, OBSERVATION_MIN, OBSERVATION_MAX)
            observation[2] = util.scale(animal.orientation, -np.pi, np.pi, OBSERVATION_MIN, OBSERVATION_MAX)
        return observation

    @property
    def is_finished(self) -> bool:
        """An Indicator showing whether the simulation is finished."""
        no_more_steps = self.current_step == self.max_steps
        all_fish_dead = self.current_fish_population == 0
        all_sharks_dead = self.current_shark_population == 0
        return no_more_steps or (all_sharks_dead and all_fish_dead)

    def __str__(self):
        banner = '\n###### Aquarium ######\n'
        env_params = {}
        signature_parameters = inspect.signature(self.__init__).parameters
        for p in signature_parameters:
            if self.__dict__.get(p) is not None:
                env_params[p] = self.__dict__.get(p)
            else:
                env_params[p] = "Init-Parameter is not part of the state dict."
        env_params.update(self.fish_types)
        env_params.update(self.shark_types)
        return banner + str(env_params)[1:-1].replace(', ', '\n')

    def render(self, draw_view_distance: bool = False):
        if self.close:
            return

        self.view.draw_background()
        all_agents = list(self.sharks) + list(self.fishes)
        for agent in all_agents:
            if self.torus:
                mirrored_position = deepcopy(agent.position)
                x_pos, y_pos = agent.position
                if x_pos > self.height - agent.radius:
                    mirrored_position += (self.width, 0)
                if x_pos < agent.radius:
                    mirrored_position += (-self.width, 0)
                if y_pos > self.width - agent.radius:
                    mirrored_position += (0, -self.height)
                if y_pos < agent.radius:
                    mirrored_position += (0, -self.height)

                self.view.draw_creature(
                    position=agent.position,
                    velocity=agent.velocity,
                    orientation=agent.orientation,
                    radius=agent.radius,
                    outer_radius=agent.view_distance,
                    color=agent.color,
                    draw_view_distance=draw_view_distance
                )

                self.view.draw_creature(
                    position=mirrored_position,
                    velocity=agent.velocity,
                    orientation=agent.orientation,
                    radius=agent.radius,
                    outer_radius=agent.view_distance,
                    color=agent.color,
                    draw_view_distance=draw_view_distance
                )
            else:
                self.view.draw_creature(
                    position=agent.position,
                    velocity=agent.velocity,
                    orientation=agent.orientation,
                    radius=agent.radius,
                    outer_radius=agent.view_distance,
                    color=agent.color,
                    draw_view_distance=draw_view_distance
                )
        self.view.render()
