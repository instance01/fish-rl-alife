#!/usr/bin/env python3
import os
import gym
from baselines import logger
from baselines import bench
# from baselines.common import tf_util as U
from baselines.ppo2 import ppo2


os.environ['OPENAI_LOGDIR'] = '/tmp'
os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'


class EnvWrapper(bench.Monitor):
    def __init__(self, env):
        self.env = env
        self.env.num_envs = 1
        bench.Monitor.__init__(
            self, self.env, 'log1.txt', allow_early_resets=True
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action.tolist()[0])
        return obs, reward, done, {}


class Experiment:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env = EnvWrapper(self.env)

    def train(self):
        logger.configure()

        model = ppo2.learn(
            network="mlp",
            env=self.env,
            total_timesteps=1000000,
            nsteps=200
        )

        import pdb; pdb.set_trace()
        for j in range(10):
            obs = self.env.reset()
            i = 0
            tot_rew = 0
            done = False
            while not done:
                i += 1
                model_inference = model.step(obs.reshape(1, -1))
                action = model_inference[0].numpy()
                obs, rewards, done, info = self.env.step(action)
                self.env.env.render()
                tot_rew += rewards
            print(tot_rew)



if __name__ == '__main__':
    Experiment().train()
