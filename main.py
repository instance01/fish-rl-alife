#!/usr/bin/env python3
import os
import sys
import json
import socket
import datetime

import tensorflow as tf
import numpy as np
from gym.core import Wrapper
from gym import spaces
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.ppo2.model import Model
from baselines.common.models import get_network_builder

# Importing network models registers them.
import network_models  # noqa
from env.aquarium import Aquarium
from env.shark import Shark
from env.fish import Fish
from custom_logger import Logger
from config import Config


os.environ['OPENAI_LOGDIR'] = 'runs/'
# os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'
# The tensorboard part is now handled by my own logger.
os.environ['OPENAI_LOG_FORMAT'] = 'stdout'


def model_inference(model, obs):
    obs = tf.cast(obs.reshape(1, -1), tf.float32)
    model_action = model.step(obs)
    return model_action[0].numpy()


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


class MultiAgentEnvWrapper(Wrapper):
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

        self.last_obs = None
        self.model = None  # Needs to be set.

        Wrapper.__init__(self, env=env)

    def step(self, action):
        sharks = list(self.env.sharks)
        if not sharks:
            # TODO .. yikes
            return ([0.] * self.n, 0, True, {})
        joint_action = {}
        for i, shark in enumerate(sharks):
            if i != 0:
                action = model_inference(self.model, self.last_obs[shark.name])
            action = (action[0][0], action[0][1], False)
            joint_action[shark.name] = action

        obs, reward, done = self.env.step(joint_action)
        self.last_obs = obs

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
        self.last_obs = obs
        return obs[shark]


class Experiment:
    def __init__(self, cfg_id, show_gui=None):
        self.cfg_id = cfg_id
        self.cfg = Config().get_cfg(cfg_id)
        print(json.dumps(self.cfg, indent=4))
        self.show_gui = self.cfg["aquarium"]["show_gui"]
        if show_gui is not None:
            self.show_gui = show_gui

        # High values increase acceleration, maximum speed and turning circle.
        Shark.FRICTION = self.cfg["aquarium"]["shark_friction"]
        # High values increase acceleration, maximum speed and turning circle.
        Shark.MAX_SPEED_CHANGE = self.cfg["aquarium"]["shark_max_speed_change"]
        # High values decrease the turning circle.
        Shark.MAX_ORIENTATION_CHANGE = float(np.radians(
            self.cfg["aquarium"]["shark_max_orientation_change"]
        ))
        Shark.VIEW_DISTANCE = self.cfg["aquarium"]["shark_view_distance"]
        Shark.PROLONGED_SURVIVAL_PER_EATEN_FISH = self.cfg["aquarium"]["shark_prolonged_survival_per_eaten_fish"]
        Shark.INITIAL_SURVIVAL_TIME = self.cfg["aquarium"]["shark_initial_survival_time"]
        Shark.PROCREATE_AFTER_N_EATEN_FISH = self.cfg["aquarium"]["shark_procreate_after_n_eaten_fish"]

        # High values decrease acceleration, maximum speed and turning circle.
        Fish.FRICTION = self.cfg["aquarium"]["fish_friction"]
        # High values increase acceleration, maximum speed and turning circle.
        Fish.MAX_SPEED_CHANGE = self.cfg["aquarium"]["fish_max_speed_change"]
        # High values decrease the turning circle.
        Fish.MAX_ORIENTATION_CHANGE = float(np.radians(
            self.cfg["aquarium"]["fish_max_orientation_change"]
        ))
        Fish.VIEW_DISTANCE = self.cfg["aquarium"]["fish_view_distance"]
        Fish.PROCREATE_AFTER_N_STEPS = self.cfg["aquarium"]["fish_procreate_after_n_steps"]

        self.env = Aquarium(
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

        if self.cfg["aquarium"]["shark_agents"] > 1:
            self.env = MultiAgentEnvWrapper(self.env)
        else:
            self.env = EnvWrapper(self.env)

    def train(self):
        hostname = socket.gethostname()
        time_str = datetime.datetime.now().strftime('%y.%m.%d-%H:%M:%S')

        self.tb_logger = Logger(self.cfg)
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
            evaluator=self.evaluate_and_log
        )

        model.save('runs/' + cfg_id + '-' + hostname + '-' + time_str + '-model')

        # import pdb; pdb.set_trace()  # noqa
        self.evaluate_and_log(model, int(total_timesteps / max_steps))

    def load_eval(self, model_filename):
        self.show_gui = True

        network = self.cfg['ppo']['network']
        ent_coef = self.cfg['ppo']['ent_coef']
        vf_coef = self.cfg['ppo']['vf_coef']
        max_grad_norm = self.cfg['ppo']['max_grad_norm']

        ob_space = self.env.observation_space
        ac_space = self.env.action_space

        policy_network_fn = get_network_builder(network)(
            num_layers=self.cfg['ppo']['num_layers'],
            num_hidden=self.cfg['ppo']['num_hidden']
        )
        network = policy_network_fn(ob_space.shape)

        model = Model(
            ac_space=ac_space,
            policy_network=network,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm
        )
        model.load(model_filename)
        self.env.model = model

        self.evaluate(model, 0)

    def evaluate(self, model, n_episode):
        """Run an evaluation game."""
        obs = self.env.reset()
        i = 0
        rewards = []
        tot_rew = 0
        while not self.env.env.is_finished:
            i += 1
            action = model_inference(model, obs)
            obs, reward, done, info = self.env.step(action)
            if self.show_gui:
                self.env.env.render()
            rewards.append(reward)
            tot_rew += reward
            if done:
                break
        print(tot_rew)
        return rewards

    def evaluate_and_log(self, model, n_episode):
        """Run an evaluation game and log to tensorboard."""
        rewards = self.evaluate(model, n_episode)
        self.tb_logger.log_summary(self.env, rewards, n_episode, prefix='Eval')


if __name__ == '__main__':
    cfg_id = sys.argv[1]
    if len(sys.argv) > 2:
        extra_action = sys.argv[2]
        if extra_action == 'det':
            from shark_baselines import get_model
            experiment = Experiment(cfg_id)
            print('TOT REW', sum(experiment.evaluate(get_model(experiment.env), 0)))
        elif extra_action == 'load':
            Experiment(cfg_id, show_gui=True).load_eval(sys.argv[3])
    else:
        Experiment(cfg_id).train()
