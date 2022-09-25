# IMPORTS
import tensorflow as tf


class DQN(tf.keras.layers.Layer):
    def __init__(self, layer_params):
        super(DQN, self).__init__()

        self.input_layer = tf.keras.layers.Dense(layer_params[0], activation='relu')
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in layer_params[1:-1]]
        self.output_layer = tf.keras.layers.Dense(layer_params[-1], activation='linear')

    def call(self, inputs):
        input = self.input_layer(inputs)
        for layer in self.hidden_layers:
            input = layer(input)

        output = self.output_layer(input)
        return output