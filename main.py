#!/usr/bin/env python3
import os
import time
import numpy as np
from gym import spaces
from custom_logger import Logger
from baselines import logger
from baselines import bench
from baselines.ppo2 import ppo2
from env.aquarium import Aquarium


os.environ['OPENAI_LOGDIR'] = 'runs/'
# os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'
# The tensorboard part is now handled by my own logger.
os.environ['OPENAI_LOG_FORMAT'] = 'stdout'


class EnvWrapper(bench.Monitor):
    def __init__(self, env):
        self.env = env
        self.env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        # spaces.Tuple((
        #     spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        #     spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        #     spaces.Discrete(2)
        # ))
        self.n = 2 + self.env.observable_sharks * 3 +\
            self.env.observable_fishes * 3 +\
            self.env.observable_walls * 2
        self.env.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n,)
        )
        self.env.reward_range = (-float('inf'), float('inf'))
        self.env.spec = None
        self.env.metadata = {'render.modes': ['human']}
        self.env.num_envs = 1
        bench.Monitor.__init__(
            self, self.env, 'log' + str(int(time.time())) + '.txt'
        )

    def step(self, action):
        sharks = list(self.env.sharks)
        if not sharks:
            # TODO .. yikes
            return ([0.] * self.n, 0, True, {})
        shark = sharks[0]
        action = (action[0][0], action[0][1], False)
        obs, reward, done = self.env.step({shark.name: action})
        shark = next(iter(done.keys()))
        return (obs.get(shark, np.array([0.] * self.n)), reward[shark], done[shark], {})

    def reset(self):
        obs = self.env.reset()
        shark = next(iter(obs.keys()))
        return obs[shark]


class Experiment:
    def __init__(self):
        self.env = Aquarium(
            nr_sharks=1,
            nr_fishes=10,
            observable_sharks=3,
            observable_fishes=3,
            observable_walls=2,
            size=30,
            max_steps=500,
            max_fish=10,
            max_sharks=1,
            torus=False,
            fish_collision=True,
            lock_screen=False,
            seed=42
        )
        self.env.select_fish_types(
            random_fish=0,
            turn_away_fish=10,
            boid_fish=0
        )
        self.env.select_shark_types(Shark_Agents=1)
        self.env = EnvWrapper(self.env)

    def train(self):
        tb_logger = Logger('cfgid')
        logger.configure()

        model = ppo2.learn(
            network="mlp",
            env=self.env,
            total_timesteps=1000000,
            nsteps=500,
            tb_logger=tb_logger
        )
        # self.env.close()

        import pdb; pdb.set_trace()

        obs = self.env.reset()
        i = 0
        tot_rew = 0
        while not self.env.env.is_finished:
            i += 1
            model_inference = model.step(obs.reshape(1, -1))
            action = model_inference[0].numpy()
            print(action)
            obs, rewards, dones, info = self.env.step(action)
            self.env.env.render()
            tot_rew += rewards
            if i % 100 == 0:
                print(tot_rew)
            if dones:
                break


if __name__ == '__main__':
    Experiment().train()
