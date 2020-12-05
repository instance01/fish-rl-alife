#!/usr/bin/env python3
import os
import sys
import json
import socket
import random
import datetime
import multiprocessing

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
        # self.env.sharks is a set, so the careful reader might lament the lack
        # of order in sets here. But this is fine. Sets still have an order,
        # it's just non-intuitive for the user - it's simply the hash order.
        # They are kept in that hash order in memory and they are returned in
        # that order. Thus, since I don't care about which shark I'm using for
        # training, as far as it's always the same one, it works out. For
        # reference, I always use the first shark returned in the set. That
        # first shark could be *any* shark from the set. But at least it's
        # always the same shark.
        # NOTE: If sharks are allowed to procreate, the assumptions do not hold
        # any longer. Adding new sharks in the middle of an episode may change
        # the order (e.g. new shark becomes the first shark in the internal
        # hash table, suddenly the shark we train with has changed).
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

        self.use_fish_pop_curriculum = \
            self.cfg['aquarium']['use_fish_pop_curriculum']
        self.use_random_fish_pop_curriculum = \
            self.cfg['aquarium']['use_random_fish_pop_curriculum']
        if self.use_fish_pop_curriculum or self.use_random_fish_pop_curriculum:
            self.fish_pop_curriculum = dict(
                (k, v) for k, v in self.cfg['aquarium']['fish_pop_curriculum']
            )

        # High values increase acceleration, maximum speed and turning circle.
        Shark.FRICTION = self.cfg["aquarium"]["shark_friction"]
        # High values increase acceleration, maximum speed and turning circle.
        Shark.MAX_SPEED_CHANGE = self.cfg["aquarium"]["shark_max_speed_change"]
        # High values decrease the turning circle.
        Shark.MAX_ORIENTATION_CHANGE = float(np.radians(
            self.cfg["aquarium"]["shark_max_orientation_change"]
        ))
        Shark.VIEW_DISTANCE = self.cfg["aquarium"]["shark_view_distance"]
        Shark.PROLONGED_SURVIVAL_PER_EATEN_FISH = \
            self.cfg["aquarium"]["shark_prolonged_survival_per_eaten_fish"]
        Shark.INITIAL_SURVIVAL_TIME = \
            self.cfg["aquarium"]["shark_initial_survival_time"]
        Shark.PROCREATE_AFTER_N_EATEN_FISH = \
            self.cfg["aquarium"]["shark_procreate_after_n_eaten_fish"]

        # High values decrease acceleration, maximum speed and turning circle.
        Fish.FRICTION = self.cfg["aquarium"]["fish_friction"]
        # High values increase acceleration, maximum speed and turning circle.
        Fish.MAX_SPEED_CHANGE = self.cfg["aquarium"]["fish_max_speed_change"]
        # High values decrease the turning circle.
        Fish.MAX_ORIENTATION_CHANGE = float(np.radians(
            self.cfg["aquarium"]["fish_max_orientation_change"]
        ))
        Fish.VIEW_DISTANCE = self.cfg["aquarium"]["fish_view_distance"]
        Fish.PROCREATE_AFTER_N_STEPS = \
            self.cfg["aquarium"]["fish_procreate_after_n_steps"]

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

    def after_epoch_cb(self, epoch):
        # Ok, the way the sausage is made here is quite fragile.
        # This assumes that the env is not 'reset' to its old cfg in any way.
        # No deepcopies, no nothing. The env was created once in the init and
        # never touched again. That's the assumption. Something to keep in
        # mind.
        if self.use_fish_pop_curriculum:
            new_fish_pop = self.fish_pop_curriculum.get(epoch, None)
            if new_fish_pop is not None:
                self.env.env.max_fish = new_fish_pop
                # TODO: I know. This is hardcoded for now. If I ever need it,
                # I'll of course add support for other fish types.
                self.env.select_fish_types(0, new_fish_pop, 0)
        if self.use_random_fish_pop_curriculum:
            idx = np.random.randint(len(self.fish_pop_curriculum))
            new_fish_pop = self.fish_pop_curriculum[idx]

            self.env.env.max_fish = new_fish_pop
            # TODO: See comment above regarding hardcoding fish type.
            self.env.select_fish_types(0, new_fish_pop, 0)

    def train(self):
        hostname = socket.gethostname()
        time_str = datetime.datetime.now().strftime('%y.%m.%d-%H:%M:%S')
        rand_str = str(int(random.random() * 100000))
        model_fname = 'runs/%s-%s-%s-%s-model' % (
            self.cfg_id,
            hostname,
            time_str,
            rand_str
        )

        self.tb_logger = Logger(self.cfg, rand_str)
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
            schedule_gamma=self.cfg['ppo']['schedule_gamma'],
            schedule_gamma_after=self.cfg['ppo']['schedule_gamma_after'],
            schedule_gamma_value=self.cfg['ppo']['schedule_gamma_value'],
            tb_logger=self.tb_logger,
            evaluator=self.evaluate_and_log,
            model_fname=model_fname,
            after_epoch_cb=self.after_epoch_cb
        )

        model.save(model_fname + '-F')  # F stands for final.

        # import pdb; pdb.set_trace()  # noqa
        self.evaluate_and_log(model, int(total_timesteps / max_steps))

        tot_rew_queue = self.tb_logger.tot_rew_queue
        return sum(tot_rew_queue) / len(tot_rew_queue), model_fname + '-0'

    def load_full(self, model_filename):
        # TODO: Honestly the `load` function further below sucks.
        # This one I like more.
        self.env.model = tf.saved_model.load(model_filename)
        return self.env.model

    def load(self, model_filename):
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
        return model

    def load_eval(self, model_filename):
        self.show_gui = True
        self.env.env.max_steps = 10000
        model = self.load(model_filename)
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
        print(i, tot_rew)
        return rewards

    def evaluate_and_log(self, model, n_episode):
        """Run an evaluation game and log to tensorboard."""
        rewards = self.evaluate(model, n_episode)
        self.tb_logger.log_summary(self.env, rewards, n_episode, prefix='Eval')

    def perturb_weights(self):
        vars = self.env.model.train_model.trainable_variables
        for var in vars:
            noise = np.random.normal(scale=.001, size=var.shape)
            var.assign_add(noise)


def worker(experiment):
    return experiment.train()


def run_evolutionary_algorithm(cfg_id):
    """Run a simple evolutionary algorithm to find excellent models.
        1. Start Experiments in 10 processes
        2. Join them
        3. Get the 5 best experiments based on running avg tot rew over 20
        4. Get the weights of the 5 best, and for each of them gen
          an Experiment with same weights
          an Experiment with weights + added noise (mutation)
        5. Go to 1.
    """
    # n_population = 10  # TODO: Not configurable right now.
    # n_top_sub_population = 5
    n_population = 2  # TODO: Not configurable right now.
    n_top_sub_population = 1
    n_generations = 2

    next_experiments = [Experiment(cfg_id) for _ in range(n_population)]

    for i in range(n_generations):
        print('GENERATION ', i + 1)
        pool = multiprocessing.Pool(processes=n_population)

        multiple_results = [
            pool.apply_async(worker, (experiment,))
            for experiment in next_experiments
        ]
        models = ([res.get() for res in multiple_results])
        models.sort(key=lambda x: x[0], reverse=True)
        pool.close()

        next_experiments = []
        for _, initial_model_fname in models[:n_top_sub_population]:
            exp1 = Experiment(cfg_id)
            exp2 = Experiment(cfg_id)
            exp1.load_full(initial_model_fname)
            exp2.load_full(initial_model_fname)
            exp2.perturb_weights()
            next_experiments.append(exp1)
            next_experiments.append(exp2)


if __name__ == '__main__':
    # python3 main.py cfg_id single  -> Train single run using cfg_id.
    # python3 main.py cfg_id multi  -> Train multiple runs using cfg_id.
    # python3 main.py cfg_id evolution  -> Train using cfg_id and evolutionary algorithm.
    # python3 main.py cfg_id [extra_action]  -> Do an extra action using cfg_id.
    #   - python3 main.py cfg_id det  -> Run deterministic shark algorithm.
    #   - python3 main.py cfg_id load runs/model1  -> Watch learnt model.
    cfg_id = sys.argv[1]
    if len(sys.argv) > 2:
        extra_action = sys.argv[2]
        if extra_action == 'det':
            from shark_baselines import get_model
            experiment = Experiment(cfg_id)
            tot_rew = sum(experiment.evaluate(get_model(experiment.env), 0))
            print('TOT REW', tot_rew)
        elif extra_action == 'load':
            Experiment(cfg_id, show_gui=True).load_eval(sys.argv[3])
        elif extra_action == 'single':
            Experiment(cfg_id).train()
        elif extra_action == 'evolution':
            run_evolutionary_algorithm(cfg_id)
        elif extra_action == 'multi':
            for _ in range(3):
                Experiment(cfg_id).train()
    else:
        # Just do 3 runs. I can cancel whenever I want.
        # Use multi_scancel.sh to cancel multiple jobs in a range.
        print('DEPRECATED! USE SINGLE OR MULTI KEYWORD!')
        for _ in range(3):
            Experiment(cfg_id).train()
