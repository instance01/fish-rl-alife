from collections import deque, namedtuple
from itertools import product
from datetime import datetime
from warnings import warn
import pandas as pd
import random
import os
import numpy as np


Transition = namedtuple('Tra'
                        'nsition',
                        ('state',
                         'action',
                         'next_state',
                         'reward')
                        )


class FelixReplayMemory:

    def __init__(self, capacity: int):
        self.replay_memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, *transition):
        self.replay_memory.append(transition)

    def get_random_sample(self, batch_size: int):
        return np.array(random.sample(self.replay_memory, batch_size))

    def size(self):
        return len(self.replay_memory)


class FabianReplayMemory(object):
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.rng = random
        self.rng.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> []:
        return self.rng.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class StatsLogger:

    def __init__(self):
        # {parameter_name, values}
        self.statistics: {str, []} = {}

    def clear(self):
        self.statistics.clear()

    def log_stats(self, **kwargs):
        for key, value in kwargs.items():
            if self.statistics.get(key) is None:
                self.statistics[key] = [value]
            else:
                self.statistics[key].append(value)

    def write_statistics(self, directory: str, file_name: str = None):

        if os.getcwd() not in os.path.abspath(directory):
            raise ValueError("The given directory path is not inside the current working directory!"
                             "Saving elsewhere is disable due to safety reasons.")

        if not os.path.exists(directory):
            os.mkdir(directory)

        if file_name is None:
            file_name =  str(datetime.now().replace(microsecond=0)) + 'statistics.csv'

        total_path = os.path.join(directory, file_name)
        data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.statistics.items()]))
        data.to_csv(total_path, index=False)


class DQNAdaptor:
    """
    This Class was create to translate a dqn decision in an environment action.
    Not the best implementation but will do for now. Feel free to update.

    """

    def __init__(self,
                 min_speed: float = -1,
                 max_speed: float = 1,
                 min_angle: float = -1,
                 max_angle: float = 1,
                 n_speed_options: int = 1,
                 n_angle_options: int = 5):

        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.n_speed_options = n_speed_options
        self.n_angle_options = n_angle_options
        self.action_map = self.__create_action_map()

    def __create_action_map(self):
        if self.n_angle_options == 1:
            turning_angles = [0]
        else:
            turning_angles = np.linspace(start=self.min_angle, stop=self.max_angle, num=self.n_angle_options)

        if self.n_speed_options == 1:
            speeds = [1]
        else:
            speeds = np.linspace(start=self.min_speed, stop=self.max_speed, num=self.n_speed_options)

        procreate = [False]
        shark_decisions = list(product(speeds, turning_angles, procreate))

        # add option to slow done
        shark_decisions += [(0, 0, False)]

        return dict(zip(range(len(shark_decisions)), shark_decisions))

    def translate(self, dqn_action):
        return self.action_map[dqn_action]

    def __str__(self):
        banner = '\n#####ACTION_SPACE#####\n'
        for item in self.action_map.items():
            banner += str(item) + '\n'
        return banner

