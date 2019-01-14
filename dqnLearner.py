import os
import socket
import time
import datetime
import fishDomain

import numpy as np
import gym

# import warnings
# warnings.filterwarnings("always")

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TestLogger

class MyTestLogger(TestLogger):
    def __init__(self):
        self.stepSum = 0
        
    def on_train_begin(self, logs):
        pass
    
    def on_episode_end(self, episode, logs):
        self.stepSum += logs['nb_steps']

    def getStepSum(self):
        return self.stepSum

fileName = ""


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)



def run(seed, steps, hiddenL, hiddenN, gamma, lr, mem, batch_size, warmup):
    testLogger = MyTestLogger()
    
    ENV_NAME = 'Fish-v0'
    
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(seed)
    env.seed(seed)
    nb_actions = env.action_space.n
    
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    for h in range(hiddenL):
        model.add(Dense(hiddenN))
        model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=mem, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, gamma=gamma, batch_size = batch_size, nb_steps_warmup=warmup, policy=policy)
    dqn.compile(Adam(lr=lr), metrics=['mae'])
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)
    
    # Load saved weights
    # time.sleep(1)
    # latestWeights = sorted(filter(lambda x: ENV_NAME in x and "dqn" in x and "h5f" in x and socket.gethostname() in x, os.listdir("weights")))[-1]
    # print latestWeights
    # time = latestWeights.split("_")[-1].replace(".h5f", "")
    # print time
#     dqn.load_weights('test/' + fileName)
    
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=500, visualize=True, verbose = 1, callbacks = [testLogger])
    stepSum = testLogger.getStepSum()
    print stepSum
    
    # After training is done, we save the final weights.
#     dqn.save_weights('test/{}_dqn_{}_weights_{}_{}_{}.h5f'.format(str(stepSum).zfill(8), ENV_NAME, socket.gethostname(), datetime.datetime.now().isoformat(), "_".join(map(str, (seed, steps, hiddenL, hiddenN, gamma, lr, mem, batch_size, warmup)))), overwrite=True)


    dqn.save_weights('weights/dqn_{}_weights_{}_{}.h5f'.format(ENV_NAME, socket.gethostname(), datetime.datetime.now().isoformat()), overwrite=True)


if __name__ == "__main__":
    for f in sorted(os.listdir("test"))[-1:]:
        fileName = f
        print f
        run(*map(num, f.replace(".h5f", "").split("_")[-9:]))



