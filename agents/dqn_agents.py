import os
import sys

import torch
import torch.nn.functional as f

from agents.utility import FabianReplayMemory
from agents.dqn_networks import SimpleDQN
from agents.utility import Transition

from projekt_parser.parser import SafeConfigParser

import inspect
from datetime import datetime


class DQN_Agent:

    def __init__(self,
                 observation_shape: [int],
                 number_of_actions: int,
                 gamma: float = 0.99,
                 epsilon: float = 1,
                 epsilon_end_at: int = 1024,
                 epsilon_min: float = 0.001,
                 mini_batch_size: int = 32,
                 warm_up_duration: int = 2000,
                 buffer_capacity: int = 50000,
                 target_update_period: int = 1000,
                 seed: int = 42) -> None:

        assert len(observation_shape) == 3, "Observation shape does not match the required dimensions!" \
                                            "Array of length three is needed."

        torch.manual_seed(seed)

        self.observation_shape = observation_shape
        self.number_of_actions = number_of_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_end_at = epsilon_end_at
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_at
        self.mini_batch_size = mini_batch_size
        self.warm_up_duration = warm_up_duration
        self.buffer_capacity = buffer_capacity
        self.target_update_period = target_update_period
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = FabianReplayMemory(buffer_capacity, seed)

        self.policy_net = SimpleDQN(observation_shape[2],
                                    observation_shape[0],
                                    observation_shape[1],
                                    number_of_actions).to(self.device)
        self.target_net = SimpleDQN(observation_shape[2],
                                    observation_shape[0],
                                    observation_shape[1],
                                    number_of_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.training_count = 0

    def mode(self, mode='train'):
        if mode == 'train':
            self.policy_net.train()
        elif mode == 'eval':
            self.policy_net.eval()
        else:
            raise NotImplementedError

    def memorize(self, state, action, next_state, reward, done):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        action = torch.tensor([action], device=self.device, dtype=torch.long)
        assert not (done and next_state is not None), "memorize done but state is not None!"
        if done:
            next_state = None
        else:
            next_state = torch.tensor([next_state], device=self.device, dtype=torch.float32)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        self.memory.push(state, action, next_state, reward)

    def train(self):
        if len(self.memory) < self.warm_up_duration:
            return
        transitions = self.memory.sample(self.mini_batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        next_state_values = torch.zeros(self.mini_batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = reward_batch + next_state_values * self.gamma

        self.optimizer.zero_grad()
        loss = f.smooth_l1_loss(state_action_values, expected_state_action_values)  # huber loss
        loss.backward()
        self.optimizer.step()
        self.training_count += 1
        if self.training_count % self.target_update_period is 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

    def policy(self, state, disable_epsilon: bool = False):
        if not disable_epsilon and torch.rand(1) < self.epsilon:
            return torch.randint(self.number_of_actions, (1,)).item()
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            return self.policy_net(state).max(1)[1].item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def reset(self, epsilon: float = 1.0):
        self.training_count = 0
        self.epsilon = epsilon
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_at
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.memory = FabianReplayMemory(self.buffer_capacity, self.seed)

    def save(self, directory: str, file_name: str = None):
        if os.getcwd() not in os.path.abspath(directory):
            raise ValueError("The given directory path is not inside the current working directory!"
                             "Saving elsewhere is disable due to safety reasons.")

        if not os.path.isdir(directory):
            os.mkdir(directory)

        if file_name is None:
            file_name = str(datetime.now().replace(microsecond=0))

        config_path = os.path.join(directory, file_name + ".ini")
        model_path = os.path.join(directory, file_name + ".pt")

        # save entire model
        torch.save(self.policy_net, model_path)

        # write config
        config = SafeConfigParser()
        init_params = {}
        # names of the all variables used in the self.__init__() function
        init_param_names = inspect.signature(self.__init__).parameters
        for p_name in init_param_names:
            p_value = self.__dict__.get(p_name)
            if p_value is None:
                print(p_name, 'could not be saved!', file=sys.stderr)
                # ToDo write to log
            else:
                init_params[p_name] = p_value

        config['AGENT'] = init_params
        config['LOAD_AGENT'] = {'model_path': model_path}
        with open(config_path, 'w') as configfile:
            config.write(configfile)

    def load_model(self, path: str):
        try:
            self.policy_net = torch.load(path)
        except RuntimeError:
            print('Model was trained using CUDA. However there is no CUDA on this mashine. '
                  'Switching to CPU.', file=sys.stderr)
            self.policy_net = torch.load(path, map_location=torch.device('cpu'))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def __str__(self):
        banner = '\n######AGENT######\n'
        init_params = inspect.signature(self.__init__).parameters
        agent_params = {p: self.__dict__.get(p) for p in init_params if self.__dict__.get(p) is not None}
        for k, v in agent_params.items():
            banner += str(k) + ': ' + str(v) + '\n'
        return banner


def load_agent(config_file_path: str):
    parser = SafeConfigParser()
    parser.read(config_file_path)

    agent_init_parameters = dict(parser.items_evaluated(section='AGENT'))
    model_path = parser.get('LOAD_AGENT', 'model_path')

    if not os.path.exists(model_path):
        raise ValueError('Path to stored model {} does not exists!'.format(model_path))

    agent = DQN_Agent(**agent_init_parameters)
    agent.load_model(model_path)
    return agent

