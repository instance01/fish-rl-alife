import tensorflow as tf
from env.animal_controller import DefaultSharkController


def get_model(env):
    class Model:
        def step(obs):
            obs = env.prepare_observation_for_controller(obs[0])
            action = DefaultSharkController.get_action(**obs)
            return tf.convert_to_tensor(
                [[[action[0], action[1]]]]
            )
    return Model
