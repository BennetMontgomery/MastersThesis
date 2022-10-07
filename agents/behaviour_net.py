# IMPORTS
import random
import math
import tensorflow as tf
import numpy as np
from agents.attention import Encoder, AttentionPooler
from agents.dqn import DQN

# CONSTANTS
VERBOSE = False

class BehaviourNet(tf.keras.Model):
    def __init__(self,
                 static_input_size, variable_input_size,
                 attention_in_d, q_layers, pooler_layers, attention_out_ff, attention_heads,
                 memory_cap, e_decay=0.001, dropout_rate=0.01):
        super(BehaviourNet, self).__init__()

        self.variable_input_size = variable_input_size
        self.static_input_size = static_input_size
        self.e_decay = e_decay

        # construct sublayers
        # encoder
        self.encoder = Encoder(attention_in_d, attention_heads, attention_out_ff, dropout_rate, variable_input_size)

        # encoding pooler
        self.pooler = AttentionPooler(attention_in_d, pooler_layers)

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
            self.memory[self.mem_counter % self.capacity] = experience

        self.mem_counter += 1

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

    def call(self, inputs: list[list[int]], training=True, attention_mask=None):
        static_input = tf.expand_dims(tf.convert_to_tensor(inputs[0], dtype='float32'), axis=0)
        dynamic_inputs = inputs[1:]

        # encode dynamic inputs
        # first dynamic input is ego observed as if it were a npc for attention
        dynamic_inputs = tf.convert_to_tensor(dynamic_inputs, dtype='float32')

        encoded = self.encoder([dynamic_inputs[0], dynamic_inputs[1:]], training=training, mask=attention_mask)

        pooled = self.pooler(encoded)

        # call q network
        if VERBOSE:
            print("[?] static_input: ", static_input)
            print("[?] encoded: ", pooled)

        q_input = tf.concat([static_input, pooled], axis=1)

        q_output = self.q_subnet(q_input)

        # return distribution of choice weighting for actions
        return q_output