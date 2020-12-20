import inspect
import numpy as np
import bisect
import itertools as it
from collections import defaultdict
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
        size=40,
        observable_sharks=2,
        observable_fishes=3,
        observable_walls=2,
        max_steps=500,
        max_fish=30,
        max_sharks=5,
        water_friction=.08,
        torus=False,
        fish_collision=True,
        lock_screen=False,
        seed=42,
        show_gui=False,
        shared_kill_zone=False,
        kill_zone_radius=10.,
        simple_kill_zone_reward=False,
        use_global_reward=False,
        stop_globally_on_first_shark_death=False,
        allow_stun_move=False,
        stun_duration_steps=100,
        stun_max_angle_diff=.4,
        stun_extend_obs=False
    ):
        # if seed is None or seed == 'none':
        #     seed = int(1000000000 * np.random.random())
        if seed is not None and seed != 'none':
            self.seed = seed
            np.random.seed(seed=seed)

        self.fishes: [Fish] = set()
        self.sharks: [Shark] = set()
        self.next_fish_id = 0
        self.next_shark_id = 0

        # Environment parameters.
        self.size = size
        self.height = self.width = size
        self.torus = torus
        self.fish_collision = fish_collision
        self.collision_space = CollisionSpace(
            self.torus,
            0,
            0,
            self.width,
            self.height
        )
        self.water_friction = water_friction
        self.use_global_reward = use_global_reward
        self.stop_globally_on_first_shark_death = stop_globally_on_first_shark_death  # noqa
        self.allow_stun_move = allow_stun_move
        self.stun_duration_steps = stun_duration_steps
        self.stun_max_angle_diff = stun_max_angle_diff
        self.stun_extend_obs = stun_extend_obs

        # Observation and action space.
        self.observable_walls = observable_walls
        self.observable_fishes = observable_fishes
        self.observable_sharks = observable_sharks
        # Diagonal line | only valid for width = height.
        self.max_animal_view_distance = np.sqrt(2) * self.width

        # Distance, angle to wall.
        self.observations_per_wall = 2
        # Distance, angle to animal and orientation of animal.
        self.observations_per_animal = 3
        if self.stun_extend_obs:
            self.observations_per_animal += 1

        # TODO Why square of observable_walls? Shouldn't it be
        # observations_per_wall?
        # own_orientation, ready_to_procreate
        self.observation_length = 2 \
            + (self.observable_walls * self.observable_walls) \
            + (self.observable_fishes + self.observable_sharks) \
            * self.observations_per_animal
        if self.stun_extend_obs:
            self.observation_length += 1

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

        # In general, let's include things we track here.
        self.dead_fishes: int = 0
        self.dead_sharks: int = 0
        self.fish_population_counter = []
        self.shark_population_counter = []
        self.shark_speed_history = []
        self.shark_tot_reward = defaultdict(int)
        # k: (shark1, shark2), v: dist
        self.shark_to_shark_dist = defaultdict(list)
        self.shark_to_shark_dist_at_kill = defaultdict(list)
        self.coop_kills = 0
        self.n_stuns = 0

        # Kill zone
        self.shared_kill_zone = shared_kill_zone
        self.kill_zone_radius = kill_zone_radius
        self.simple_kill_zone_reward = simple_kill_zone_reward

        # GUI
        self.show_gui = show_gui
        self.close = lock_screen
        self.screen_height = self.screen_width = 600
        if self.show_gui:
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

    def select_fish_types(self, random_fish=0, turn_away_fish=0, boid_fish=0):
        self.fish_types['RandomFish'] = random_fish
        self.fish_types['TurnAwayFish'] = turn_away_fish
        self.fish_types['BoidFish'] = boid_fish
        return self

    def select_shark_types(self, shark_agents=0):
        # TODO: Update the docstring below.
        """Derzeit gibt es not keine Möglichkeit, statische und normale Agent zu kombinieren
           Das kommt noch"""
        #self.shark_types['RandomShark'] = Random_Shark
        #self.shark_types['DefaultShark'] = Default_Shark
        self.shark_types['shark_agents'] = shark_agents
        return self

    def reset(self) -> np.array:
        self.current_step = 0

        self.next_shark_id = 0
        self.next_fish_id = 0

        self.sharks = set()
        self.fishes = set()

        self.dead_sharks = 0
        self.dead_fishes = 0
        self.fish_population_counter = []
        self.shark_population_counter = []
        self.shark_speed_history = []
        self.shark_tot_reward = defaultdict(int)
        self.shark_to_shark_dist = defaultdict(list)
        self.shark_to_shark_dist_at_kill = defaultdict(list)
        self.coop_kills = 0

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

        self.fish_population_counter.append(len(self.fishes))
        self.shark_population_counter.append(len(self.sharks))
        for (s1, s2) in it.combinations(list(self.track_shark_reward.keys()), 2):
            observation = self.observe_animal(s1, s2)
            self.shark_to_shark_dist[(s1.name(), s2.name())].append(observation[0])

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
            "random_shark": RandomShark,
            "default_shark": DefaultShark,
            "shark_agents": SharkAgent
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
            if self.show_gui:
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

        self.fish_population_counter.append(len(self.fishes))
        self.shark_population_counter.append(len(self.sharks))
        # TODO: This is roughly copy paste from reset() (~l.205)
        for (s1, s2) in it.combinations(list(self.track_shark_reward.keys()), 2):
            key = (s1.name(), s2.name())
            observation = self.observe_animal(s1, s2)
            self.shark_to_shark_dist[key].append(observation[0])
            if self.track_shark_reward[s1] > 0 or self.track_shark_reward[s2] > 0:
                self.shark_to_shark_dist_at_kill[key].append(observation[0])

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

    def _on_shark_fish_collision(self, shark, fish):
        if fish in self.fishes:
            self.fishes.remove(fish)
        if shark in self.track_shark_reward:
            # This was to check whether we have real cooperation or
            # herding.
            # Well, we have herding.
            #
            # for shark_ in self.sharks:
            #     if shark_ == shark:
            #         print('s', shark_.starving_indicator())
            #     else:
            #         print('o', shark_.starving_indicator())
            # print('')

            if self.shared_kill_zone:
                # This may look like it works for multiple sharks, but no.
                # Only works for two sharks.
                # TODO: Support multiple sharks at some point.
                reward_main_shark = 10.

                for shark_ in self.sharks:
                    if shark_ == shark:
                        continue
                    observation = self.observe_animal(shark_, fish)
                    dist = observation[0]

                    # TODO This is hacky.
                    min_dist = shark_.radius + fish.radius
                    max_dist = min(shark_.view_distance, self.max_animal_view_distance)
                    radius_dist = util.scale(self.kill_zone_radius, min_dist, max_dist, 0, OBSERVATION_MAX)

                    print(dist, radius_dist)
                    if dist > radius_dist:
                        continue
                    # Seems like another shark participated in the kill.
                    # They now have to share the reward based on the distance of
                    # the other shark.

                    self.coop_kills += 1

                    if self.simple_kill_zone_reward:
                        reward_curr_shark = 5.
                        reward_main_shark = 5.
                    else:
                        ratio = 1. - dist / radius_dist
                        print(ratio)
                        # TODO: To simplify, could consider integer only rewards.
                        # E.g.: round(ratio * 5)
                        # Right now it's floats.
                        reward_curr_shark = ratio * 5
                        reward_main_shark = 10. - reward_curr_shark

                    self.track_shark_reward[shark_] += reward_curr_shark
                    self.shark_tot_reward[shark_] += reward_curr_shark

                self.track_shark_reward[shark] += reward_main_shark
                self.shark_tot_reward[shark] += reward_main_shark
            else:
                self.track_shark_reward[shark] += 10
                self.shark_tot_reward[shark] += 10

                if self.use_global_reward:
                    for shark_ in self.sharks:
                        if shark_ == shark:
                            continue
                        self.track_shark_reward[shark_] += 10
                        self.shark_tot_reward[shark_] += 10

            self.dead_fishes += 1
            shark.eaten_fish += 1

    def _handle_stun_move(self, a1, a2):
        """Make a shark unable to move for a certain number of steps and turn it
        purple, if another shark hits it at a certain angle."""
        # TODO Maybe make sure the hitting shark has to have a certain velocity.
        # Else sometimes they may just touch randomly and stun themselves.
        dx, dy = a2.position - a1.position
        _, direction = util.cartesian_to_polar(dx, dy)
        if abs(direction - a1.orientation) < self.stun_max_angle_diff:
            a2.color = Shark.STUN_COLOR
            a2.stun_steps = self.stun_duration_steps
            self.n_stuns += 1

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
                # TODO: Hacky.
                # Decrease speed by 99% when swimming backwards.
                if speed < 0:
                    speed *= .01
                    angle *= .01

                if shark.stun_steps > 0:
                    shark.stun_steps -= 1
                    speed = 0
                    angle = 0
                    if shark.stun_steps == 0:
                        shark.color = Shark.COLOR
                shark.act(speed, angle)
                shark.survived_n_steps += 1

                self.shark_speed_history.append(speed)

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
                    self._on_shark_fish_collision(shark, fish)
                else:
                    fish.survived_n_steps += 1

        # Solve shark-shark-collisions.
        if self.fish_collision and self.sharks:
            combinations = it.combinations(self.sharks, 2)
            for a1, a2 in combinations:
                if self.collision_space.check_collision(a1, a2):
                    if self.allow_stun_move:
                        self._handle_stun_move(a1, a2)
                        self._handle_stun_move(a2, a1)
                    self.collision_space.perform_collision(a1, a2)

        if self.stop_globally_on_first_shark_death and len(starved_sharks) > 0:
            self.sharks = set()

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
        if self.stun_extend_obs:
            observer_data.append(int(observer.stun_steps != 0))

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
            if self.stun_extend_obs:
                observation[3] = int(animal.stun_steps != 0)
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
        if self.close or not self.show_gui:
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
