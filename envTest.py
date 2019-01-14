import gym
from fishDomain.envs.fishDomain import FishDomainEnv

env = FishDomainEnv(10, 2)
# env = gym.make('Fish-v0')
env.seed(2)
 
for i_episode in range(1000):
    observation = env.reset()  # reset for each new trial
    for t in range(10000):  # run for 100 timesteps or until done, whichever is first
        env.render()
#         action = env.action_space.sample()  # select a random action (see https://github.com/openai/gym/wiki/CartPole-v0)
        action = 2
        observation, reward, done, info = env.step(action)
#         print observation
#         print reward
#         print
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

