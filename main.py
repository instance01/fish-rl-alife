#!/usr/bin/env python3
import os
import sys
import json

import numpy as np
from gym.core import Wrapper
from gym import spaces
from baselines import logger
from baselines.ppo2 import ppo2

from env.aquarium import Aquarium
from custom_logger import Logger
from config import Config


os.environ['OPENAI_LOGDIR'] = 'runs/'
# os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'
# The tensorboard part is now handled by my own logger.
os.environ['OPENAI_LOG_FORMAT'] = 'stdout'


class EnvWrapper(Wrapper):
    def __init__(self, env):
        self.env = env
        self.env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
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
        Wrapper.__init__(self, env=env)

    def step(self, action):
        sharks = list(self.env.sharks)
        if not sharks:
            # TODO .. yikes
            return ([0.] * self.n, 0, True, {})
        shark = sharks[0]
        action = (action[0][0], action[0][1], False)
        obs, reward, done = self.env.step({shark.name: action})
        shark = next(iter(done.keys()))
        return (
            obs.get(shark, np.array([0.] * self.n)),
            reward[shark],
            done[shark],
            {}
        )

    def reset(self):
        obs = self.env.reset()
        shark = next(iter(obs.keys()))
        return obs[shark]


class Experiment:
    def __init__(self, cfg_id):
        self.cfg = Config().get_cfg(cfg_id)
        print(json.dumps(self.cfg, indent=4))
        self.show_gui = self.cfg["aquarium"]["show_gui"]

        self.env = Aquarium(
            nr_sharks=self.cfg["aquarium"]["nr_sharks"],
            nr_fishes=self.cfg["aquarium"]["nr_fishes"],
            observable_sharks=self.cfg["aquarium"]["observable_sharks"],
            observable_fishes=self.cfg["aquarium"]["observable_fishes"],
            observable_walls=self.cfg["aquarium"]["observable_walls"],
            size=self.cfg["aquarium"]["size"],
            max_steps=self.cfg["aquarium"]["max_steps"],
            max_fish=self.cfg["aquarium"]["max_fish"],
            max_sharks=self.cfg["aquarium"]["max_sharks"],
            torus=self.cfg["aquarium"]["torus"],
            fish_collision=self.cfg["aquarium"]["fish_collision"],
            lock_screen=self.cfg["aquarium"]["lock_screen"],
            seed=self.cfg["aquarium"]["seed"],
            show_gui=self.show_gui
        )
        self.env.select_fish_types(
            self.cfg["aquarium"]["random_fish"],
            self.cfg["aquarium"]["turn_away_fish"],
            self.cfg["aquarium"]["boid_fish"]
        )
        self.env.select_shark_types(
            self.cfg["aquarium"]["shark_agents"]
        )

        self.env = EnvWrapper(self.env)

    def train(self):
        self.tb_logger = Logger(self.cfg['cfg_id'])
        logger.configure()

        total_timesteps = self.cfg['ppo']['total_timesteps']
        max_steps = self.cfg['aquarium']['max_steps']

        model = ppo2.learn(
            env=self.env,
            network=self.cfg['ppo']['network'],
            total_timesteps=total_timesteps,
            # TODO Seed..
            # seed=self.cfg['ppo']['seed'],
            # TODO: For now for consistency we use Aquarium max_steps as nsteps
            nsteps=max_steps,
            ent_coef=self.cfg['ppo']['ent_coef'],
            lr=self.cfg['ppo']['lr'],
            vf_coef=self.cfg['ppo']['vf_coef'],
            max_grad_norm=self.cfg['ppo']['max_grad_norm'],
            gamma=self.cfg['ppo']['gamma'],
            lam=self.cfg['ppo']['lam'],
            log_interval=self.cfg['ppo']['log_interval'],
            nminibatches=self.cfg['ppo']['nminibatches'],
            noptepochs=self.cfg['ppo']['noptepochs'],
            cliprange=self.cfg['ppo']['cliprange'],
            save_interval=self.cfg['ppo']['save_interval'],
            num_layers=self.cfg['ppo']['num_layers'],
            num_hidden=self.cfg['ppo']['num_hidden'],
            tb_logger=self.tb_logger,
            evaluator=self.evaluate
        )

        # import pdb; pdb.set_trace()  # noqa
        self.evaluate(model, int(total_timesteps / max_steps))

    def evaluate(self, model, n_episode):
        obs = self.env.reset()
        i = 0
        rewards = []
        tot_rew = 0
        while not self.env.env.is_finished:
            i += 1
            model_inference = model.step(obs.reshape(1, -1))
            action = model_inference[0].numpy()
            obs, reward, done, info = self.env.step(action)
            if self.show_gui:
                self.env.env.render()
            rewards.append(reward)
            tot_rew += reward
            if done:
                break
        print(tot_rew)
        self.tb_logger.log_summary(self.env, rewards, n_episode, prefix='Eval')


if __name__ == '__main__':
    cfg_id = sys.argv[1]
    Experiment(cfg_id).train()
