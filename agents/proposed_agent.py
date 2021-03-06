# IMPORTS
import libsumo
import gym
import tensorflow as tf
from agent import Agent
from roundabout_gym import roundabout_env

'''
    BehaviourNet outputs a behaviour decision from the following:
        - follow leader (car directly in front, or accelerate to speed limit if no front car)
        - choose turn
        - change lane
'''
class BehaviourNet(tf.keras.Model):
    def __init__(self, memory_size, layers):
        super(BehaviourNet, self).__init__()
        self.memory_size = memory_size
        self.layers = layers

        # Building input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(memory_size,))

        # Building fully connected intermediate layers
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in layers]

        # Building output layer:
        #   0: change lane left
        #   1: change lane right
        #   1: choose turn left
        #   2: choose turn right
        #   3: follow leader
        self.output_layer = tf.keras.layers.Dense(3, activation='linear')

'''
    Brake/Throttle network takes in the current state space and a behaviour code generated by BehaviourNet and 
    converts it to an appropriate brake/speed value
'''
class BTNet(tf.keras.Model):
    def __init__(self, state_size, behaviour_code_num, layers):
        super(BTNet, self).__init__()
        self.memory_size = state_size + behaviour_code_num
        self.layers = layers

        # Building input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.memory_size,))

        # Building hidden layers
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in layers]

class ProposedAgent(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False, VIEW_DISTANCE=30):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose, VIEW_DISTANCE)