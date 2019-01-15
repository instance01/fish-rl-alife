import os
import socket
import sys
import datetime
from fishDomain.envs.fishDomain import FishDomainEnv
from naturalSorting import naturalKeys

import numpy as np

# import warnings
# warnings.filterwarnings("always")

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *
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


def createAgent(seed, steps, hiddenL, hiddenNActor, hiddenNCritic, gamma, lr, mem, batch_size, warmup, numberActions, observationShape):
    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + observationShape))
    for h in range(hiddenL):
        actor.add(Dense(hiddenNActor))
        actor.add(Activation('relu'))
    actor.add(Dense(numberActions))
    actor.add(Activation('linear'))
    print(actor.summary())
    
    action_input = Input(shape=(numberActions,), name='action_input')
    observation_input = Input(shape=(1,) + observationShape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    for h in range(hiddenL):
        x = Dense(hiddenNCritic)(x)
        x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())
    
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=mem, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=numberActions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=numberActions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=warmup, nb_steps_warmup_actor=warmup,
                      random_process=random_process, gamma=gamma, target_model_update=1e-3, batch_size=batch_size)
    agent.compile(Adam(lr=lr, clipnorm=1.), metrics=['mae'])
    return agent



def run(seed, steps, hiddenL, hiddenNActor, hiddenNCritic, gamma, lr, mem, batch_size, warmup, fish, obsFish):
    
    
    testLogger = MyTestLogger()
    ENV_NAME = 'Fish-v1'
    
    # Get the environment and extract the number of actions.
    env = FishDomainEnv("ddpg", fish, obsFish, True)
    np.random.seed(seed)
    env.seed(seed)
    
    ddpg = createAgent(seed, steps, hiddenL, hiddenNActor, hiddenNCritic, gamma, lr, mem, batch_size, warmup, env.action_space.shape[0], env.observation_space.shape)
    
    # Load saved weights
    # time.sleep(1)
    # latestWeights = sorted(filter(lambda x: ENV_NAME in x and "ddpg" in x and "h5f" in x and socket.gethostname() in x, os.listdir("weights")))[-1]
    # print latestWeights
    # time = latestWeights.split("_")[-1].replace(".h5f", "")
    # print time
    if not learn:
        ddpg.load_weights('weights/' + fileName)
    
    
    dqnStatic = createAgent(seed, steps, hiddenL, hiddenNActor, hiddenNCritic, gamma, lr, mem, batch_size, warmup, env.action_space.shape[0], env.observation_space.shape)
    dqnStatic.actor = clone_model(ddpg.actor)
    dqnStatic.training = False
    env.setAgents(ddpg, dqnStatic)
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    if learn:
        ddpg.fit(env, nb_steps=steps, visualize=True, verbose=1)
    
    
    # Finally, evaluate our algorithm for 5 episodes.
    ddpg.test(env, nb_episodes = 1000 if learn else 5, visualize= False if learn else True, verbose=1, callbacks=[testLogger])
    stepSum = testLogger.getStepSum()
    print stepSum
    
    # After training is done, we save the final weights.
    if learn:
        ddpg.save_weights('weights/{}_ddpg_{}_weights_{}_{}_{}.h5f'.format(str(stepSum).zfill(12), ENV_NAME, socket.gethostname(), datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), "_".join(map(str, (seed, steps, hiddenL, hiddenNActor, hiddenNCritic, gamma, lr, mem, batch_size, warmup, fish, obsFish)))), overwrite=True)


#     ddpg.save_weights('weights/dqn_{}_weights_{}_{}.h5f'.format(ENV_NAME, socket.gethostname(), datetime.datetime.now().isoformat()), overwrite=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        learn = True
        run(*map(num, sys.argv[1:]))
    else:
        learn = False
        for f in [x for x in sorted(os.listdir("weights"), key=naturalKeys) if "ddpg" in x]:
            print f
        for f in [x for x in sorted(os.listdir("weights"), key=naturalKeys) if "ddpg" in x][int(sys.argv[1])::2]:
            fileName = f.replace("_actor", "").replace("_critic", "")
            print "Using:"
            print f
            run(*map(num, f.replace(".h5f", "").split("_")[-13:-1]))



