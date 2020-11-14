import os
import numpy as np

from agents.dqn_agents import load_agent
from agents.utility import DQNAdaptor
from agents.utility import StatsLogger

from env.aquarium import Aquarium

from collections import deque

from projekt_parser.parser import SafeConfigParser


if __name__ == '__main__':

    n_episodes = 100
    experiment = "2020-04-14 15:29:56_95"

    root = "./runs/"
    config = '/experiment_config.ini'
    checkpoint = 10_000
    phase = 'two'

    path = root + experiment + config
    assert os.path.exists(path)

    # ====================== create env ============================================================================

    env_params = {'size': 30,
                  'observable_walls': 2,
                  'observable_sharks': 3,
                  'observable_fishes': 3,
                  'max_steps': 500,
                  'max_fish': 10,
                  'max_sharks': 1,
                  'seed': 42,
                  'torus': False,
                  'fish_collision': True,
                  'lock_screen': False,
                  }

    env = Aquarium(**env_params)
    env.select_fish_types(Random_Fish=0, Turn_Away_Fish=1, Boid_Fish=0)
    env.select_shark_types(Shark_Agent=0)
    print(env)


    # ====================== create agent ==========================================================================

    config = SafeConfigParser()
    config.read(path)
    option = "phase_{}_checkpoint_{}".format(phase, checkpoint)

    agent_ini_files = config.get_evaluated(section='CHECKPOINTS', option=option)
    agent = load_agent(agent_ini_files)
    print(agent)

    # ====================== action_space ==========================================================================

    action_map = DQNAdaptor()
    print(action_map, '\n' * 3)

    # ====================== logging ===============================================================================

    logger = StatsLogger()

    # ====================== trace length ==========================================================================

    trace_len = config.get_evaluated(section='HISTORY', option='trace_length')

    # ====================== start =================================================================================

    for training_episode in range(1, n_episodes + 1):

        joint_shark_observation = env.reset()

        observation_history = deque([[0] * env.observation_length for _ in range(trace_len)],
                                    maxlen=trace_len)

        new_observation_history = deque([[0] * env.observation_length for _ in range(trace_len)],
                                        maxlen=trace_len)

        # ====================== begin episode =========================================================================

        episode_reward = 0
        while not env.is_finished:

            # ToDo: this only works with one shark !!!!!!
            assert len(joint_shark_observation) == 1

            # shark act
            joint_shark_action = {}
            shark_dqn_action = {}
            # select_random_actors = np.random.permutation(list(joint_shark_observation.keys()))
            for shark in joint_shark_observation.keys():
                observation_history.append(joint_shark_observation[shark])
                action = agent.policy(np.asarray(observation_history).flatten())
                shark_dqn_action[shark] = action
                joint_shark_action[shark] = action_map.translate(action)


            # update observation
            new_state = env.step(joint_shark_action)
            new_shark_observation, shark_reward, shark_done = new_state

            episode_reward += sum(shark_reward.values())
            joint_shark_observation = new_shark_observation
            env.render()
