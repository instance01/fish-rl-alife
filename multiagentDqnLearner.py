import os
import socket
import sys
import datetime
from fishDomain.envs.fishDomain import FishDomainEnv
from naturalSorting import naturalKeys

import numpy as np

# import warnings
# warnings.filterwarnings("always")

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TestLogger

class MyTestLogger(TestLogger):
    def __init__(self):
        self.stepSum = 0
        
    def on_train_begin(self, logs):
        pass
    
    def on_episode_end(self, episode, logs):
        self.stepSum += logs['episode_reward']

    def getStepSum(self):
        return self.stepSum

fileName = ""
learn = None

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def createAgent(seed, steps, hiddenL, hiddenN, gamma, lr, mem, batch_size, warmup, numberActions, observationShape):
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + observationShape))
    for h in range(hiddenL):
        model.add(Dense(hiddenN))
        model.add(Activation('relu'))
    model.add(Dense(numberActions))
    model.add(Activation('linear'))
    print(model.summary())
    
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=mem, window_length=1)
    policy = EpsGreedyQPolicy()
    agent = DQNAgent(model=model, nb_actions=numberActions, memory=memory, gamma=gamma, batch_size=batch_size, nb_steps_warmup=warmup, policy=policy)
    agent.compile(Adam(lr=lr), metrics=['mae'])
    return agent


def run(seed, steps, hiddenL, hiddenN, gamma, lr, mem, batch_size, warmup, fish, obsFish):
    testLogger = MyTestLogger()
    ENV_NAME = 'Fish-v0'
    
    # Get the environment and extract the number of actions.
    env = FishDomainEnv("dqn", fish, obsFish)
    np.random.seed(seed)
    env.seed(seed)
    
    dqn = createAgent(seed, steps, hiddenL, hiddenN, gamma, lr, mem, batch_size, warmup, env.action_space.n, env.observation_space.shape)
    
    # Load saved weights
    # time.sleep(1)
    # latestWeights = sorted(filter(lambda x: ENV_NAME in x and "dqn" in x and "h5f" in x and socket.gethostname() in x, os.listdir("weights")))[-1]
    # print latestWeights
    # time = latestWeights.split("_")[-1].replace(".h5f", "")
    # print time
    if not learn:
        dqn.load_weights('weights/' + fileName)
    
    
    dqnStatic = createAgent(seed, steps, hiddenL, hiddenN, gamma, lr, mem, batch_size, warmup, env.action_space.n, env.observation_space.shape)
    dqnStatic.model.set_weights(dqn.model.get_weights())
    dqnStatic.training = False
    env.setAgents(dqn, dqnStatic)
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    if learn:
        dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)
    
    
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=1000 if learn else 5, visualize= False if learn else True, verbose=1, callbacks=[testLogger])
    stepSum = testLogger.getStepSum()
    print stepSum
    
    # After training is done, we save the final weights.
    if learn:
        dqn.save_weights('weights/{}_dqn_{}_weights_{}_{}_{}.h5f'.format(str(stepSum).zfill(12), ENV_NAME, socket.gethostname(), datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), "_".join(map(str, (seed, steps, hiddenL, hiddenN, gamma, lr, mem, batch_size, warmup, fish, obsFish)))), overwrite=True)


#     dqn.save_weights('weights/dqn_{}_weights_{}_{}.h5f'.format(ENV_NAME, socket.gethostname(), datetime.datetime.now().isoformat()), overwrite=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        learn = True
        run(*map(num, sys.argv[1:]))
    else:
        learn = False
        for f in [x for x in sorted(os.listdir("weights"), key=naturalKeys) if "dqn" in x]:
            print f
        for f in [x for x in sorted(os.listdir("weights"), key=naturalKeys) if "dqn" in x][int(sys.argv[1]):]:
            fileName = f
            print "Using:"
            print f
            run(*map(num, f.replace(".h5f", "").split("_")[-11:]))



