import os
import sys
import socket

import numpy as np

import logging
from datetime import datetime
from collections import deque

from agents.utility import StatsLogger
from agents.dqn_agents import load_agent, DQN_Agent
from agents.utility import DQNAdaptor

from projekt_parser.parser import SafeConfigParser

from env.aquarium import Aquarium
from env.fish import Fish


class Experiment:

    def __init__(self,
                 config_file_path: str):

        self.config_file_path = config_file_path
        self.config = SafeConfigParser()
        self.config.read(config_file_path)

        self.main_folder: str
        self.checkpoint_folder: str
        self.evaluation_folder: str
        self.not_done_file: str
        self.logger: logging
        self.__build_folders_structure()

        self.env: Aquarium
        self.agent: DQN_Agent
        self.trace_len: int
        self.action_map: DQNAdaptor
        self.training_episodes: int
        self.create_checkpoint_every: int
        self.__build_experiment()

        self.started_at = datetime.now()

    def __build_folders_structure(self):

        save_here = self.config['EXPERIMENT']['save_here']

        # create main folder
        experiment_id = str(np.random.randint(0, 100))
        folder_name = str(datetime.now().replace(microsecond=0)) + '_' + experiment_id
        self.main_folder = os.path.join(save_here, folder_name)
        os.mkdir(self.main_folder)

        # logging
        log_file = os.path.join(self.main_folder, 'output.log')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(log_file))

        # indicate whether the experiment is done
        # file will be removed if the experiment is done
        self.not_done_file = os.path.join(self.main_folder, 'not_done.txt')
        with open(self.not_done_file, 'w') as file:
            file.write(socket.gethostname())

        # save models here
        self.checkpoint_folder = os.path.join(self.main_folder, 'checkpoints')
        os.mkdir(self.checkpoint_folder)

        # save stats here
        self.stats_folder = os.path.join(self.main_folder, 'statistics')
        os.mkdir(self.stats_folder)

        # save eval files here
        self.evaluation_folder = os.path.join(self.main_folder, 'evaluation')
        os.mkdir(self.evaluation_folder)

    def __build_experiment(self):

        # ====================== create env ============================================================================
        self.logger.info('building experiment')
        env_init_params = self.config.items_evaluated(section='ENVIRONMENT')
        fish_controller = self.config.items_evaluated(section='FISH_CONTROLLER')
        env = Aquarium(**env_init_params)
        env.select_fish_types(*fish_controller.values())

        self.env = env
        self.logger.info(self.env)

        # ====================== trace length ==========================================================================

        self.trace_len = self.config.get_evaluated(section='HISTORY', option='trace_length')

        # ====================== create agent ==========================================================================

        agent_init_params = self.config.items_evaluated('AGENT')
        agent_init_params['observation_shape'] = [1, 1, env.observation_length * self.trace_len]
        self.agent = DQN_Agent(**agent_init_params)
        self.logger.info(self.agent)

        # ====================== action_space ==========================================================================

        self.action_map = DQNAdaptor(min_angle=-0.33,
                                     max_angle=0.33,
                                     n_speed_options=1,
                                     n_angle_options=5)
        self.logger.info(self.action_map)

        # ====================== episodes ==============================================================================

        self.training_episodes = self.config.get_evaluated(section='EXPERIMENT', option='training_episodes')
        self.create_checkpoint_every = self.config.get_evaluated(section='EXPERIMENT', option='checkpoint_every')
        self.evaluation_episodes = self.config.get_evaluated(section='EXPERIMENT', option='evaluation_episodes')

    def start_phase_one(self):

        self.logger.info('\nstart phase one\n')

        # ====================== logging ===============================================================================

        collect_stats = StatsLogger()
        checkpoints = {}

        # ====================== setup fish ============================================================================

        Fish.PROCREATE_AFTER_N_POINTS = 1024
        Fish.MAX_SPEED_CHANGE = 0.0
        delta_speed_change = 0.005
        max_speed_change = 0.04

        # ====================== start =================================================================================

        for training_episode in range(1, self.training_episodes + 1):

            # increase fish speed every 1000
            if not training_episode % 1000 and training_episode > self.agent.epsilon_end_at:
                Fish.MAX_SPEED_CHANGE = min(Fish.MAX_SPEED_CHANGE + delta_speed_change,
                                            max_speed_change)

            # random amount of targets
            self.env.nr_fishes = np.random.randint(1, 11)
            joint_shark_observation = self.env.reset()

            observation_history = deque([[0] * self.env.observation_length for _ in range(self.trace_len)],
                                        maxlen=self.trace_len)

            new_observation_history = deque([[0] * self.env.observation_length for _ in range(self.trace_len)],
                                            maxlen=self.trace_len)

        # ====================== begin episode =========================================================================

            episode_reward = 0
            while not self.env.is_finished():

                # ToDo: this only works with one shark !!!!!!
                assert len(joint_shark_observation) == 1

                # shark act
                joint_shark_action = {}
                shark_dqn_action = {}
                # select_random_actors = np.random.permutation(list(joint_shark_observation.keys()))
                for shark in joint_shark_observation.keys():
                    observation_history.append(joint_shark_observation[shark])
                    action = self.agent.policy(np.asarray(observation_history).flatten())
                    shark_dqn_action[shark] = action
                    joint_shark_action[shark] = self.action_map.translate(action)

                # update observation
                new_state = self.env.step(joint_shark_action)
                new_joint_shark_observation, shark_reward, shark_done = new_state

                for shark in new_joint_shark_observation.keys():
                    new_observation_history.append(new_joint_shark_observation[shark])
                    new_state = np.asarray(new_observation_history).flatten()
                    state = np.asarray(observation_history).flatten()
                    self.agent.memorize(state=state,
                                        action=shark_dqn_action[shark],
                                        next_state=new_state,
                                        reward=shark_reward[shark],
                                        done=shark_done[shark],
                                        )

                self.agent.train()
                episode_reward += sum(shark_reward.values())
                joint_shark_observation = new_joint_shark_observation
                #self.env.render()

        # ========================= log run stats ======================================================================

            self.agent.decay_epsilon()
            collect_stats.log_stats(episodes=training_episode,
                                    episode_rewards=episode_reward,
                                    epsilons=self.agent.epsilon,
                                    killed_fish=self.env.killed_fishes,
                                    nr_fishes=self.env.nr_fishes,
                                    fish_speed=Fish.MAX_SPEED_CHANGE,
                                    )

            if self.create_checkpoint_every and not training_episode % self.create_checkpoint_every:
                checkpoint_name = 'phase_one_checkpoint_{}'.format(training_episode)
                self.agent.save(directory=self.checkpoint_folder, file_name=checkpoint_name)
                checkpoints[checkpoint_name] = os.path.join(self.checkpoint_folder, checkpoint_name + '.ini')

            self.logger.info("phase one >>> training agent: {} episode: {}, reward: {:.2f}, nr_fishes {}".
                             format('sharks', training_episode, episode_reward, self.env.nr_fishes))

        # ========================== save stats ========================================================================

        collect_stats.write_statistics(self.stats_folder, file_name='phase_one_stats.csv')

        if self.config.has_section(section='CHECKPOINTS'):
            previous_checkpoints = self.config.items_evaluated(section='CHECKPOINTS')
            checkpoints.update(previous_checkpoints)
        self.config['CHECKPOINTS'] = checkpoints

        # =========================== end of run =======================================================================

        # save an updated copy of experiment
        with open(os.path.join(self.main_folder, 'experiment_config.ini'), 'w') as copy_config:
            self.config.write(copy_config)

        self.__evaluate_phase_one()

    def __evaluate_phase_one(self):

        if not self.evaluation_episodes:
            return

        checkpoints = self.config.items_evaluated(section='CHECKPOINTS')
        for checkpoint, agent_ini in checkpoints.items():

            if 'phase_one' not in checkpoint:
                return

            self.logger.info('\nstart eval at checkpoint\n{}\n -epsilon disabled\n'.format(checkpoint))
            agent = load_agent(agent_ini)

            # ====================== logging ===========================================================================

            collect_stats = StatsLogger()

            # ====================== setup fish ========================================================================

            Fish.PROCREATE_AFTER_N_POINTS = 1024
            Fish.MAX_SPEED_CHANGE = 0.04
            self.env.nr_fishes = 1

            # ====================== start =============================================================================

            for training_episode in range(1, self.evaluation_episodes + 1):

                joint_shark_observation = self.env.reset()

                observation_history = deque([[0] * self.env.observation_length for _ in range(self.trace_len)],
                                            maxlen=self.trace_len)

            # ====================== begin episode =================================================================

                episode_reward = 0
                while not self.env.is_finished():

                    # ToDo: this only works with one shark !!!!!!
                    assert len(joint_shark_observation) == 1

                    # shark act
                    joint_shark_action = {}
                    shark_dqn_action = {}
                    # select_random_actors = np.random.permutation(list(joint_shark_observation.keys()))
                    for shark in joint_shark_observation.keys():
                        observation_history.append(joint_shark_observation[shark])
                        action = agent.policy(np.asarray(observation_history).flatten(), disable_epsilon=True)
                        shark_dqn_action[shark] = action
                        joint_shark_action[shark] = self.action_map.translate(action)

                    # update observation
                    new_state = self.env.step(joint_shark_action)
                    new_shark_observation, shark_reward, shark_done = new_state

                    episode_reward += sum(shark_reward.values())
                    joint_shark_observation = new_shark_observation
                    #self.env.render()

            # ========================= log run stats ==================================================================

                collect_stats.log_stats(episodes=training_episode,
                                        episode_rewards=episode_reward,
                                        epsilons=agent.epsilon,
                                        killed_fish=self.env.killed_fishes,
                                        nr_fishes=self.env.nr_fishes,
                                        procreations=self.env.fish_procreated,
                                        )

                self.logger.info("evaluating phase one >>> {} episode: {}, reward: {:.2f}, nr_fishes {}".
                                 format(checkpoint, training_episode, episode_reward, self.env.nr_fishes))

            # ========================== save stats ====================================================================

            collect_stats.write_statistics(self.evaluation_folder, file_name='eval_{}.csv'.
                                    format(checkpoint))

    def start_phase_two(self):

        self.logger.info('\nstarting phase two')

        # ====================== logging ===============================================================================

        collect_stats = StatsLogger()
        checkpoints = {}

        # ====================== setup fish ============================================================================

        Fish.MAX_SPEED_CHANGE = 0.04
        Fish.PROCREATE_AFTER_N_POINTS = 100

        # ====================== start =================================================================================

        self.agent.reset(epsilon=1.0)
        self.logger.info(self.agent)

        for training_episode in range(1, self.training_episodes + 1):

            # random amount of targets
            self.env.nr_fishes = np.random.randint(1, 4)
            joint_shark_observation = self.env.reset()

            observation_history = deque([[0] * self.env.observation_length for _ in range(self.trace_len)],
                                        maxlen=self.trace_len)

            new_observation_history = deque([[0] * self.env.observation_length for _ in range(self.trace_len)],
                                            maxlen=self.trace_len)

        # ====================== begin episode =========================================================================

            episode_reward = 0
            while not self.env.is_finished():

                # ToDo: this only works with one shark !!!!!!
                assert len(joint_shark_observation) == 1

                # shark act
                joint_shark_action = {}
                shark_dqn_action = {}
                # select_random_actors = np.random.permutation(list(joint_shark_observation.keys()))
                for shark in joint_shark_observation.keys():
                    observation_history.append(joint_shark_observation[shark])
                    action = self.agent.policy(np.asarray(observation_history).flatten())
                    shark_dqn_action[shark] = action
                    joint_shark_action[shark] = self.action_map.translate(action)

                # update observation
                new_state = self.env.step(joint_shark_action)
                new_joint_shark_observation, shark_reward, shark_done = new_state

                for shark in new_joint_shark_observation.keys():
                    new_observation_history.append(new_joint_shark_observation[shark])
                    new_state = np.asarray(new_observation_history).flatten()
                    state = np.asarray(observation_history).flatten()
                    self.agent.memorize(state=state,
                                        action=shark_dqn_action[shark],
                                        next_state=new_state,
                                        reward=shark_reward[shark],
                                        done=shark_done[shark],
                                        )

                self.agent.train()
                episode_reward += sum(shark_reward.values())
                joint_shark_observation = new_joint_shark_observation
                #self.env.render()

        # ========================= log run stats ======================================================================

            self.agent.decay_epsilon()
            collect_stats.log_stats(episodes=training_episode,
                                    episode_rewards=episode_reward,
                                    epsilons=self.agent.epsilon,
                                    killed_fish=self.env.killed_fishes,
                                    nr_fishes=self.env.nr_fishes,
                                    )

            if self.create_checkpoint_every and not training_episode % self.create_checkpoint_every:
                checkpoint_name = 'phase_two_checkpoint_{}'.format(training_episode)
                self.agent.save(directory=self.checkpoint_folder, file_name=checkpoint_name)
                checkpoints[checkpoint_name] = os.path.join(self.checkpoint_folder, checkpoint_name + '.ini')

            self.logger.info("phase two >>> training agent: {} episode: {}, reward: {:.2f}, nr_fishes {}".
                             format('sharks', training_episode, episode_reward, self.env.nr_fishes))

        # ========================== save stats ========================================================================

        collect_stats.write_statistics(self.stats_folder, file_name='phase_two_stats.csv')

        if self.config.has_section(section='CHECKPOINTS'):
            previous_checkpoints = self.config.items_evaluated(section='CHECKPOINTS')
            checkpoints.update(previous_checkpoints)
        self.config['CHECKPOINTS'] = checkpoints

        # =========================== end of run =======================================================================

        # save an updated copy of experiment
        with open(os.path.join(self.main_folder, 'experiment_config.ini'), 'w') as copy_config:
            self.config.write(copy_config)

        self.__evaluate_phase_two()

    def __evaluate_phase_two(self):

        if not self.evaluation_episodes:
            return

        checkpoints = self.config.items_evaluated(section='CHECKPOINTS')
        for checkpoint, agent_ini in checkpoints.items():

            if 'phase_two' not in checkpoint:
                return

            self.logger.info('\nstart eval at checkpoint\n{}\n -epsilon disabled\n'.format(checkpoint))
            agent = load_agent(agent_ini)

            # ====================== logging ===========================================================================

            collect_stats = StatsLogger()

            # ====================== setup fish ========================================================================

            Fish.MAX_SPEED_CHANGE = 0.04
            Fish.PROCREATE_AFTER_N_POINTS = 100
            self.env.nr_fishes = 1

            # ====================== start =============================================================================

            for training_episode in range(1, self.evaluation_episodes + 1):

                joint_shark_observation = self.env.reset()

                observation_history = deque([[0] * self.env.observation_length for _ in range(self.trace_len)],
                                            maxlen=self.trace_len)

            # ====================== begin episode =================================================================

                episode_reward = 0
                while not self.env.is_finished():

                    # ToDo: this only works with one shark !!!!!!
                    assert len(joint_shark_observation) == 1

                    # shark act
                    joint_shark_action = {}
                    shark_dqn_action = {}
                    # select_random_actors = np.random.permutation(list(joint_shark_observation.keys()))
                    for shark in joint_shark_observation.keys():
                        observation_history.append(joint_shark_observation[shark])
                        action = agent.policy(np.asarray(observation_history).flatten(), disable_epsilon=True)
                        shark_dqn_action[shark] = action
                        joint_shark_action[shark] = self.action_map.translate(action)

                    # update observation
                    new_state = self.env.step(joint_shark_action)
                    new_shark_observation, shark_reward, shark_done = new_state

                    episode_reward += sum(shark_reward.values())
                    joint_shark_observation = new_shark_observation
                    #self.env.render()

            # ========================= log run stats ==================================================================

                collect_stats.log_stats(episodes=training_episode,
                                        episode_rewards=episode_reward,
                                        epsilons=agent.epsilon,
                                        killed_fish=self.env.killed_fishes,
                                        nr_fishes=self.env.nr_fishes,
                                        procreations=self.env.fish_procreated,
                                       )

                self.logger.info("evaluating phase two >>> {} episode: {}, reward: {:.2f}, nr_fishes {}".
                                 format(checkpoint, training_episode, episode_reward, self.env.nr_fishes))

            # ========================== save stats ====================================================================

            collect_stats.write_statistics(self.evaluation_folder, file_name='eval_{}.csv'.
                                    format(checkpoint))

    def flag_done(self):
        with open(os.path.join(self.main_folder, 'profiling.txt'), 'w') as file:
            file.write('machine: {}, duration: {}'.format(socket.gethostname(), datetime.now() - self.started_at))
        os.remove(self.not_done_file)
        self.logger.info('\nExperiment successfully done.\n')
