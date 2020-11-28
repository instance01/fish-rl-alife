import numpy as np
import tensorflow as tf
from baselines.a2c.utils import ortho_init
from baselines.common.models import register


def mlp_norm(norm, num_layers=2, num_hidden=64, activation=tf.tanh):
    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))

        x_input = tf.keras.Input(shape=input_shape)
        h = x_input

        for i in range(num_layers):
            h = tf.keras.layers.Dense(
                units=num_hidden,
                kernel_initializer=ortho_init(np.sqrt(2)),
                name='mlp_fc{}'.format(i),
                activation=activation
            )(h)
            h = norm(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network

    return network_fn


@register("mlp_batchnorm")
def mlp_batchnorm(num_layers=2, num_hidden=64, activation=tf.tanh):
    return mlp_norm(
        tf.keras.layers.BatchNormalization,
        num_layers,
        num_hidden,
        activation
    )


@register("mlp_layernorm")
def mlp_layernorm(num_layers=2, num_hidden=64, activation=tf.tanh):
    return mlp_norm(
        tf.keras.layers.LayerNormalization,
        num_layers,
        num_hidden,
        activation
    )
