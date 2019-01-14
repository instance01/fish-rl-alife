from gym.envs.registration import register

register(
    id='Fish-v0',
    entry_point='fishDomain.envs:FishDomainEnv',
)