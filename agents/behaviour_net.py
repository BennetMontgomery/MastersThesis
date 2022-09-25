# IMPORTS
import random
import math
import tensorflow as tf
import numpy as np
from attention import Encoder, Embedder
from dqn import DQN


class BehaviourNet(tf.keras.Model):
    def __init__(self, static_input_size, variable_input_size, attention_in_d, q_layers, memory_cap, e_decay=0.001, dropout_rate=0.1):
        super(BehaviourNet, self).__init__()

        self.variable_input_size = variable_input_size
        self.static_input_size = static_input_size
        self.e_decay = e_decay

        # construct sublayers
        # embedding layer
        self.tokenizer = Embedder(variable_input_size, attention_in_d)

        # encoder layer
        self.encoder = Encoder(attention_in_d, 8, q_layers[0], dropout_rate)

        # Q subnet
        self.q_subnet = DQN(q_layers)

        # build memory buffer
        self.memory_cap = memory_cap
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

    def select_action(self, obs, step):
        # get the probabiltiy of selecting a random action instead of e-greedy from policy
        rate = math.exp(-1 * step * self.e_decay)

        if rate > random.random():
            # return random action, exploration rate at current step, indicator that action was random
            return random.randrange(3), rate, False
        else:
            # return argmax_a q(s, a), rate at current step, indicator that action was not random
            return np.argmax(self(obs)), rate, True

    @tf.function
    def call(self, inputs: list[list[int]], **kwargs):
        static_input = inputs[0]
        dynamic_inputs = inputs[1:]

        # embed dynamic inputs
        dynamic_inputs = [tf.convert_to_tensor(dynamic_input) for dynamic_input in dynamic_inputs]
        for



        # static_input = inputs[0]
        # dynamic_inputs = inputs[1:]
        #
        # # call phi networks
        # # ASSUMPTION: dynamic inputs are passed in the same order as their equivalent phi networks
        # phi_input = self.phi_networks[0][0](dynamic_inputs[0])
        # for hidden_phi_layer in self.phi_networks[0][1]:
        #     phi_input = hidden_phi_layer(phi_input)
        #
        # phi_outputs = phi_input
        # phi_network = 0
        # for input in range(1, len(dynamic_inputs)):
        #     # call matching phi network to generate vector for pooling
        #     # if we've reached a new type of input, switch to next phi network type
        #     if len(dynamic_inputs[input]) != len(dynamic_inputs[input - 1]):
        #         phi_network += 1
        #
        #     phi_input = self.phi_networks[phi_network][0](dynamic_inputs[input])
        #     for hidden_phi_layer in self.phi_networks[0][1]:
        #         phi_input = hidden_phi_layer(phi_input)
        #
        #     # pool phi outputs
        #     phi_outputs = tf.add(phi_outputs, phi_input)
        #
        # # pass to rho
        # rho_input = self.rho_input_layer(phi_outputs)
        # rho_output = self.rho_output_layer(rho_input)
        #
        # # pass rho as dynamic input vector to Q
        # q_input = self.input_layer(rho_input)
        #
        # for layer in self.hidden_layers:
        #     q_input = layer(q_input)
        #
        # output = self.output_layer(q_input)
        # return output