# IMPORTS
import tensorflow as tf
import numpy as np
import random
import math
from agents.agent import Agent
from roundabout_gym.roundabout_env import RoundaboutEnv
from collections import namedtuple
from datetime import datetime
from agents.behaviour_net import BehaviourNet
from agents.dqn import DQN

# CONSTANTS
SAVE_MODEL = False

class ProposedAgent(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False, VIEW_DISTANCE=30):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose, VIEW_DISTANCE)

        # predeclared variables
        self.behaviour_net = None
        self.throttle_net = None

    def flatten_obs(self, env_obs):
        # flatten observations to something passable to tensorflow
        obs = [[env_obs["number"], env_obs["speed"], env_obs["accel"], env_obs["laneid"],
                           env_obs["lanepos"], env_obs["dti"], env_obs["lanes"][0], env_obs["lanes"][1]]]

        if "vehicles" in env_obs.keys():
            for vehicle in env_obs["vehicles"]:
                obs.append([vehicle[keya][keyb] for keya in vehicle.keys() for keyb in vehicle[keya]])

        # lights = []
        #
        # if "lights" in env_obs.keys():
        #     for light in env_obs["lights"]:
        #         lights.append(np.array([light[key] for key in light.keys()]))

        # return static vector followed by dynamic array observation vector
        return obs

    def train_nets(self):
        # UNIVERSAL HYPERPARAMS
        batch_size = 32 # replay memory sample size
        gamma = 0.9 # reward discount factor
        update_freq_t = 25 # number of rounds between updates to the target q tnet
        replay_cap = 8000 # maximum number of experience objects to store in memory
        episodes = 2000 # rounds of training
        log_freq = 1 # number of rounds between printing of loss and other ML statistics
        alpha = 0.001 # learning rate
        optimizer = tf.optimizers.Adam(alpha)

        # BEHAVIOUR SPECIFIC HYPERPARAMS
        b_action_space_size = 3
        b_q_layers = [256, 128, b_action_space_size] # layer parameters in the behavioural q network
        update_freq_b = 25 # number of rounds between updates to the target q bnet
        num_heads_b = 8 # number of attention heads
        encoder_layer_b = 256 # width of the attention embeddings
        encoder_feed_forward_b = 128 # width of the attention post-processing network
        pooler_layers_b = [128] # width and depth of the seq-2-vec processing network
        e_decay_b = 0.001 # epsilon decay for random action selection

        # THROTTLE SPECIFIC HYPERPARAMS
        t_action_space_size = 200
        t_q_layers = [256, 128, t_action_space_size]  # layer parameters in the throttle q network
        update_freq_t = 25  # number of rounds between updates to the target q tnet
        num_heads_t = 8  # number of attention heads
        encoder_layer_t = 256  # width of the attention embeddings
        encoder_feed_forward_t = 128  # width of the attention post-processing network
        pooler_layers_t = [128]  # width and depth of the seq-2-vec processing network
        e_decay_t = 0.001  # epsilon decay for random action selection

        # start environment
        self.test_env = RoundaboutEnv(self)

        # declare replay buffer memory format
        Experience = namedtuple('Experience', ['states', 'actions', 'reward', 'state_primes', 'terminates'])

        # CREATE NETWORKS
        self.behaviour_net = BehaviourNet(
            static_input_size=8,
            variable_input_size=4,
            attention_heads=num_heads_b,
            q_layers=b_q_layers,
            pooler_layers=pooler_layers_b,
            attention_out_ff=encoder_feed_forward_b,
            attention_in_d=encoder_layer_b,
            memory_cap=replay_cap,
            e_decay=e_decay_b
        )

        self.target_behaviour_net = BehaviourNet(
            static_input_size=8,
            variable_input_size=4,
            attention_heads=num_heads_b,
            q_layers=b_q_layers,
            pooler_layers=pooler_layers_b,
            attention_out_ff=encoder_feed_forward_b,
            attention_in_d=encoder_layer_b,
            memory_cap=replay_cap,
            e_decay=e_decay_b
        )

        self.throttle_net = BehaviourNet(
            static_input_size=8,
            variable_input_size=4,
            attention_heads=num_heads_t,
            q_layers=t_q_layers,
            pooler_layers=pooler_layers_t,
            attention_out_ff=encoder_feed_forward_t,
            attention_in_d=encoder_layer_t,
            memory_cap=replay_cap,
            e_decay=e_decay_t
        )

        self.target_throttle_net = BehaviourNet(
            static_input_size=8+1,
            variable_input_size=4,
            attention_heads=num_heads_t,
            q_layers=t_q_layers,
            pooler_layers=pooler_layers_t,
            attention_out_ff=encoder_feed_forward_t,
            attention_in_d=encoder_layer_t,
            memory_cap=replay_cap,
            e_decay=e_decay_t
        )

        # ensure starting equality of networks
        policy_params = self.behaviour_net.trainable_variables
        target_params = self.target_behaviour_net.trainable_variables

        for pvar, tvar in zip(policy_params, target_params):
            tvar.assign(pvar.numpy())

        policy_params = self.throttle_net.trainable_variables
        target_params = self.target_throttle_net.trainable_variables

        for pvar, tvar in zip(policy_params, target_params):
            tvar.assign(pvar.numpy())

        # track reward history
        reward_history = np.empty(episodes)

        # train
        for episode in range(episodes):
            self.test_env.reset()

            # generic training variables
            episode_return = 0
            step = 0
            b_loss_history = []
            t_loss_history = []
            terminated = False

            while not terminated:
                obs = self.test_env.sample()
                step += 1

                # select behaviour according to e-greedy policy
                b_action, _, _ = self.behaviour_net.select_action(obs, step)
                # add selected behaviour to observation to pass to throttle net
                t_obs = obs.copy()
                t_obs[0].append(b_action)

                # select throttle action
                t_action, _, _ = self.throttle_net.select_action(obs, step)

                # apply action to environment
                state_prime, reward, terminated = self.test_env.step([b_action, t_action])
                episode_return += reward

                # store experience in replay buffers
                self.throttle_net.add_mem(Experience(t_obs, t_action, reward, state_prime, terminated))
                self.behaviour_net.add_mem(Experience(obs, b_action, reward, state_prime, terminated))

                # replay buffer sampling
                if self.throttle_net.mem_counter > batch_size:
                    behaviour_memories = self.behaviour_net.sample_replay_batch(batch_size)
                    throttle_memories = self.behaviour_net.sample_replay_batch(batch_size)
                    behaviour_minibatch = Experience(*zip(*behaviour_memories))
                    throttle_minibatch = Experience(*zip(*throttle_memories))

                    behaviour_states = np.asarray(behaviour_minibatch[0])
                    behaviour_actions = np.asarray(behaviour_minibatch[1])
                    behaviour_rewards = np.asarray(behaviour_minibatch[2])
                    behaviour_state_primes = np.asarray(behaviour_minibatch[3])
                    behaviour_terminates = np.asarray(behaviour_minibatch[4])

                    throttle_states = np.asarray(throttle_minibatch[0])
                    throttle_actions = np.asarray(throttle_minibatch[1])
                    throttle_rewards = np.asarray(throttle_minibatch[2])
                    throttle_state_primes = np.asarray(throttle_minibatch[3])
                    throttle_terminates = np.asarray(behaviour_minibatch[4])

                    # UPDATE BEHAVIOUR NETWORK
                    # calculate loss and apply gradient descent
                    q_prime = np.max(self.target_behaviour_net(np.atleast_2d(behaviour_state_primes)), axis=1)
                    q_optimal = np.where(behaviour_terminates, behaviour_rewards, behaviour_rewards + gamma*q_prime)
                    q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                    with tf.GradientTape() as tape:
                        q = tf.math.reduce_sum(
                            self.behaviour_net(np.atleast_2d(self.behaviour_net(behaviour_states)).astype('float32')) *
                                               tf.one_hot(behaviour_actions, b_action_space_size), axis=1)
                        loss = tf.math.reduce_mean(tf.square(tf.square(q_optimal - q)))

                    # Update using ADAM
                    variables = self.behaviour_net.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables))

                    # record loss
                    b_loss_history.append(loss.numpy())
                else:
                    b_loss_history.append(0)
                    t_loss_history.append(0)


    def select_action(self, time_step):
        # actions may not be selected without training
        if (self.throttle_net is None) or (self.behaviour_net is None):
            raise RuntimeError("[!!] Throttle net and Behaviour net must be trained or loaded before calling select_action.")