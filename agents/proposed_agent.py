# IMPORTS
import tensorflow as tf
import random
from agent import Agent
from roundabout_gym.roundabout_env import RoundaboutEnv
from collections import namedtuple
from datetime import datetime

# CONSTANTS
SAVE_MODEL = False

'''
    BehaviourNet outputs a behaviour decision from the following:
        - follow leader (car directly in front, or accelerate to speed limit if no front car)
        - choose turn
        - change lane
'''
class BehaviourNet(tf.keras.Model):
    def __init__(self, static_input_size, variable_input_sizes, phi_layers, q_layers, memory_cap):
        # ensure valid layer sizes
        assert(len(phi_layers) > 1)

        super(BehaviourNet, self).__init__()
        # Network construction parameters
        # Phi network construction
        self.phi_networks = []
        for object_type in variable_input_sizes:
            self.phi_networks.append(
                (
                    tf.keras.layers.InputLayer(input_shape=(object_type,)),
                    [tf.keras.layers.Dense(layer, activation='relu') for layer in phi_layers]
                )
            )

        # Rho network construction
        self.rho_input_layer = tf.keras.layers.InputLayer(input_shape=(phi_layers[-1],))
        self.rho_output_layer = tf.keras.layers.Dense(phi_layers[-2], activation='relu')

        # Q network
        # Building input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(static_input_size + phi_layers[-2],))

        # Building fully connected intermediate layers
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in q_layers]

        # Building output layer:
        #   0: change lane left
        #   1: change lane right
        #   2: follow leader
        self.output_layer = tf.keras.layers.Dense(3, activation='linear')

        # Replay buffer parameters
        self.memory_cap = memory_cap

        # instantiate buffer
        self.memory = []
        self.mem_counter = 0

    def add_mem(self, experience):
        # fill to memory cap
        if len(self.memory) < self.memory_cap:
            self.memory.append(experience)
        # or replace a pseudo-random memory if cap reached
        else:
            self.memory[self.counter % self.capacity] = experience

        self.counter += 1

    def sample_replay_batch(self, batch_size):
        if len(self.memory) > batch_size:
            return random.sample(self.memory, batch_size)
        else:
            raise ValueError("[!!] Replay Buffer queried before {batch} memories accumulated".format(batch=batch_size))

    @tf.function
    def call(self, inputs, **kwargs):
        static_input = inputs[0]
        dynamic_inputs = inputs[1:]

        # call phi networks
        phi_outputs = []
        for input in dynamic_inputs:
            # call matching phi network to generate vector for pooling


        for layer in self.hidden_layers:
            input = layer(input)

        output = self.output_layer(input)
        return output


'''
    Brake/Throttle network takes in the current state space and a behaviour code generated by BehaviourNet and 
    converts it to an appropriate brake/speed value
'''
class BTNet(BehaviourNet):
    def __init__(self, static_state_size, variable_state_sizes, psi_layers, q_layers, memory_cap):
        super(BTNet, self).__init__(static_state_size + 1, variable_state_sizes, psi_layers, q_layers, memory_cap)
        # phi and rho are built in super init call

        # Building q input layer. There is one extra input for behaviour token ID
        self.q_input_layer = tf.keras.layers.InputLayer(input_shape=(psi_layers[-2] + static_state_size + 1,))

        # Building hidden layers
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in q_layers]

        # Building output layer
        self.output_layer = tf.keras.layers.Dense(22, activation='linear')


class ProposedAgent(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False, VIEW_DISTANCE=30):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose, VIEW_DISTANCE)

        self.behaviour_net = None
        self.throttle_net = None

    def train_nets(self):
        # HYPERPARAMS
        batch_size = 32
        gamma = 0.9
        update_freq = 25
        replay_cap = 8000
        episodes = 2000
        log_freq = 1
        alpha = 0.001
        optimizer = tf.optimizers.Adam(alpha)
        b_layers = [200, 100, 100]
        t_layers = [200, 100, 100]

        # start environment
        test_env = RoundaboutEnv(self)

        # declare replay buffer experience format
        Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'state_primes', 'terminates'])

        # create policy networks and target networks with equivalent starting parameters
        # self.behaviour_net = BehaviourNet()



    def select_action(self, time_step):
        pass