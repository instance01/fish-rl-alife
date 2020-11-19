import pickle
import threading
import socket
import random
import datetime
from collections import defaultdict

import tensorflow as tf


lock = threading.Lock()


class Logger:
    def __init__(self, cfg_id):
        _id = "-".join([
            datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S"),
            str(int(random.random() * 1000)),
            socket.gethostname(),
            cfg_id
        ])
        self.logdir = "runs/" + _id
        self.writer = tf.summary.create_file_writer(self.logdir)
        self.writer.set_as_default()
        self.custom_metrics = defaultdict(list)
        self.custom_file = self.logdir + "/custom.pickle"

    def log_summary(self, env, rewards, n_episode, prefix='Train'):
        tf.summary.scalar(prefix + '/Tot_Reward', sum(rewards), n_episode)
        tf.summary.scalar(prefix + '/Dead_Fishes', env.dead_fishes, n_episode)
        tf.summary.scalar(prefix + '/Dead_Sharks', env.dead_sharks, n_episode)
        self.log_file(prefix + '/Fish_Population', env.fish_population_counter)
        self.log_file(prefix + '/Shark_Population', env.shark_population_counter)

    def log_kv(self, k, v, step):
        tf.summary.scalar(k, v, step)

    def log_file(self, key, arr):
        # TODO Here we'll just log to a file for later parsing for matplotlib
        # or pgfplots. E.g. with matplotlib just use meshgrid (see axiros wiki)
        # Since I don't know how PPO runs environments in parallel, let's make
        # this thread safe in general.
        with lock:
            self.custom_metrics[key].append(arr)
            with open(self.custom_file, 'wb+') as f:
                pickle.dump(self.custom_metrics, f, protocol=3)
