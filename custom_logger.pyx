#cython: language_level=3, boundscheck=False
import random
import threading
import socket
import datetime
from itertools import combinations
from collections import defaultdict

import msgpack
import tensorflow as tf


lock = threading.Lock()


class Logger:
    def __init__(self, cfg, rand_str, evolution=False, runs_folder='runs'):
        _id = "-".join([
            datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S"),
            rand_str,
            socket.gethostname(),
            cfg['cfg_id']
        ])
        if evolution:
            _id += '-evolution'
        self.logdir = runs_folder + "/" + _id
        self.writer = tf.summary.create_file_writer(self.logdir)
        self.custom_metrics = defaultdict(list)
        self.custom_file = self.logdir + "/custom.msgpack"
        with self.writer.as_default():
            tf.summary.text('Info/Params', str(cfg), 0)

        self.tot_rew_queue = []  # TODO Make it an actual queue?

    def log_summary(self, env, rewards, n_episode, prefix='Train'):
        fish_pop = env.fish_population_counter
        shark_pop = env.shark_population_counter
        tot_rew = sum(rewards)
        self.tot_rew_queue.append(tot_rew)
        if len(self.tot_rew_queue) > 100:
            self.tot_rew_queue.pop(0)
        with self.writer.as_default():
            tf.summary.scalar(prefix + '/Tot_Reward', tot_rew, n_episode)
            tf.summary.scalar(prefix + '/Dead_Fishes', env.dead_fishes, n_episode)
            tf.summary.scalar(prefix + '/Dead_Sharks', env.dead_sharks, n_episode)
            tf.summary.scalar(prefix + '/Last_Fish_Population', fish_pop[-1], n_episode)
            tf.summary.scalar(prefix + '/Last_Shark_Population', shark_pop[-1], n_episode)
            tf.summary.scalar(prefix + '/Coop_Kills', env.coop_kills, n_episode)
            # print('fc', env.full_coop_kills)
            if env.dead_fishes > 0:
                coop_kills_ratio = env.coop_kills / env.dead_fishes
                full_coop_kills_ratio = env.full_coop_kills / env.dead_fishes
            else:
                coop_kills_ratio = 0
                full_coop_kills_ratio = 0
            tf.summary.scalar(prefix + '/Coop_Kills_Ratio', coop_kills_ratio, n_episode)
            tf.summary.scalar(prefix + '/Full_Coop_Kills_Ratio', full_coop_kills_ratio, n_episode)
            tf.summary.scalar(prefix + '/Stuns', env.n_stuns, n_episode)
            tf.summary.histogram(prefix + '/Last_Fish_Population_H', fish_pop, n_episode)
            # TODO: January 10 - Got rid of shark pop histogram. Fuck that.
            # tf.summary.histogram(prefix + '/Last_Shark_Population_H', shark_pop, n_episode)
            tf.summary.histogram(prefix + '/Shark_Speed_H', env.shark_speed_history, n_episode)
            # msgpack is nice but whatever, let's disable this. We need speed.
            # self.log_file(prefix + '/Fish_Population', fish_pop)
            # self.log_file(prefix + '/Shark_Population', shark_pop)
            for i, (_, tot_reward) in enumerate(env.shark_tot_reward.items()):
                name = prefix + '/Sharks/Shark%d_Tot_Reward' % i
                tf.summary.scalar(name, tot_reward, n_episode)

            # TODO: January 10 - Got rid of distances. Fuck that.
            # # Let's not keep distances at evaluation phase..
            # if prefix != 'Eval':
            #     for (s1, s2) in combinations(list(env.shark_tot_reward.keys()), 2):
            #         key = (s1.name(), s2.name())
            #         name_dist = 'Sharks/Shark-%s-%s_Dist_To_Dist' % key
            #         name_dist_at_kill = 'Sharks/Shark-%s-%s_Dist_To_Dist_At_Kill' % key
            #         tf.summary.histogram(
            #             name_dist, env.shark_to_shark_dist[key], n_episode
            #         )
            #         tf.summary.histogram(
            #             name_dist_at_kill, env.shark_to_shark_dist_at_kill[key], n_episode
            #         )

    def log_kv(self, k, v, step):
        # Used by the PPO algorithm internally to log things like policy
        # entropy etc.
        with self.writer.as_default():
            tf.summary.scalar(k, v, step)

    def log_file(self, key, arr):
        # TODO Here we'll just log to a file for later parsing for matplotlib
        # or pgfplots. E.g. with matplotlib just use meshgrid (see axiros wiki)
        # Since I don't know how PPO runs environments in parallel, let's make
        # this thread safe in general.
        with lock:
            self.custom_metrics[key].append(arr)
            with open(self.custom_file, 'wb+') as f:
                # pickle.dump(self.custom_metrics, f, protocol=3)
                packed = msgpack.packb(self.custom_metrics)
                f.write(packed)


class EvolutionLogger:
    def __init__(self, cfg_id):
        rand_str = str(int(random.random() * 100e6))
        _id = "-".join([
            datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S"),
            rand_str,
            socket.gethostname(),
            cfg_id,
            'evolution-meta'
        ])
        # Here, the runs folder is hard coded. Since I don't use the EA anymore.
        self.logdir = "runs/" + _id
        self.writer = tf.summary.create_file_writer(self.logdir)

    def log(self, rew_model_list, n_generation):
        with self.writer.as_default():
            tf.summary.text('Info', str(rew_model_list), n_generation)
            tot_rews = [tot_rew for tot_rew, _ in rew_model_list]
            tf.summary.histogram('Tot_Reward', tot_rews, n_generation)
