#!/usr/bin/env python3
from baselines import logger
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
from env.aquarium import Aquarium


class Experiment:
    def __init__(self):
        self.num_timesteps = 10000

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
            lock_screen=True,
            seed=42
        )
        self.env.select_fish_types(
            RandomFish=0,
            TurnAwayFish=10,
            BoidFish=0
        )
        self.env.select_shark_types(Shark_Agents=1)

    def train(self):
        logger.configure()

        U.make_session(num_cpu=1).__enter__()

        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(
                name=name,
                ob_space=ob_space,
                ac_space=ac_space,
                hid_size=64,
                num_hid_layers=2
            )

        pposgd_simple.learn(
            self.env, policy_fn, max_timesteps=self.num_timesteps,
            timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear'
        )
        self.env.close()


if __name__ == '__main__':
    Experiment().train()
