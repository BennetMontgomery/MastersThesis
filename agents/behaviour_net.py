# IMPORTS
import random
import math
import tensorflow as tf
import numpy as np
from agents.attention import Encoder, AttentionPooler
from agents.dqn import DQN
from parameters.simulator_params import maximum_npcs
from agents.replay_manager import ReplayManager

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
        self.attention_in_d = attention_in_d
        self.e_decay = e_decay
        self.q_layers = q_layers

        # construct sublayers
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(maximum_npcs+1,static_input_size))

        # encoder
        self.encoder = Encoder(attention_in_d, attention_heads, attention_out_ff, dropout_rate, variable_input_size)

        # encoding pooler
        self.pooler = AttentionPooler(attention_in_d, pooler_layers, static_input_size)

        # static embedder
        self.static_embedder = tf.keras.layers.Embedding(200, attention_in_d, mask_zero=True)

        # Q subnet
        self.q_subnet = DQN(q_layers, input_s=((static_input_size*2)*attention_in_d,))

        # build memory buffer
        self.replay_manager = ReplayManager(replay_cap=memory_cap)

    def add_mem(self, experience):
        self.replay_manager.add_mem(experience)

    def sample_replay_batch(self, batch_size):
        return self.replay_manager.sample_batch(batch_size)

    def select_action(self, obs, step):
        # get the probabilitiy of selecting a random action instead of e-greedy from policy
        rate = math.exp(-1 * step * self.e_decay)

        # convert obs to tensor
        try:
            obs = tf.convert_to_tensor(obs, dtype='float32')
        except:
            for l in obs:
                print(l)
            exit(1)

        if rate > random.random():
            # return random action, exploration rate at current step, indicator that action was random
            return random.randrange(self.q_layers[-1]), rate, False
        else:
            # return argmax_a q(s, a), rate at current step, indicator that action was not random
            # print("selecting action without failure")
            return np.argmax(self(obs)), rate, True

    def call(self, inputs, training=True, attention_mask=None):
        # if pre-batched
        inputs = tf.convert_to_tensor(inputs, dtype='float32')

        if tf.rank(inputs) < 3:
            inputs = tf.expand_dims(inputs, axis=0)

        static_input = tf.convert_to_tensor([obs[0] for obs in inputs])
        dynamic_inputs = tf.convert_to_tensor([obs[1:] for obs in inputs])

        static_input = self.static_embedder(static_input)

        # encode dynamic inputs
        encoded =self.encoder(dynamic_inputs, training=training, mask=attention_mask)

        pooled = self.pooler(encoded)

        # call q network
        if VERBOSE:
            print("[?] static_input: ", static_input)
            print("[?] encoded: ", pooled)

        q_input = tf.reshape(tf.concat([static_input, pooled], axis=1), [tf.shape(static_input)[0], ((self.static_input_size)*2)*self.attention_in_d])

        if VERBOSE:
            print("[?] q_input: ", q_input)

        q_output = self.q_subnet(q_input)

        # return distribution of choice weighting for actions
        return q_output