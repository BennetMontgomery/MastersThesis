# IMPORTS
import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, layer_params, input_s):
        super(DQN, self).__init__()

        self.input_s = input_s

        self.submodel = tf.keras.Sequential([tf.keras.layers.Dense(layer_params[0], input_shape=input_s, activation='relu')])
        for layer in layer_params[1:-1]:
            self.submodel.add(tf.keras.layers.Dense(layer, activation='relu'))
        self.submodel.add(tf.keras.layers.Dense(layer_params[-1], activation='linear'))

        # self.input_layer = tf.keras.layers.Dense(layer_params[0], input_shape=input_s, activation='relu')
        # self.hidden_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in layer_params[1:-1]]
        # self.output_layer = tf.keras.layers.Dense(layer_params[-1], activation='linear')

    def call(self, inputs):
        # inputs = tf.expand_dims(tf.reshape(inputs, [shape for shape in self.input_s]),axis=0)

        output = self.submodel(inputs)
        return output

        # input = self.input_layer(inputs)
        # for layer in self.hidden_layers:
        #     input = layer(input)
        #
        # output = self.output_layer(input)
        # return output