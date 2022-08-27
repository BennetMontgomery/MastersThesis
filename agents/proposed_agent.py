# IMPORTS
import tensorflow as tf
import numpy as np
import random
import math
from agent import Agent
from roundabout_gym.roundabout_env import RoundaboutEnv
from collections import namedtuple
from datetime import datetime
from behaviour_net import BehaviourNet

# CONSTANTS
SAVE_MODEL = False


'''
    Brake/Throttle network takes in the current state space and a behaviour code generated by BehaviourNet and 
    converts it to an appropriate brake/speed value
'''
class BTNet(BehaviourNet):
    def __init__(self, static_state_size, variable_state_sizes, phi_layers, q_layers, memory_cap):
        super(BTNet, self).__init__(static_state_size + 1, variable_state_sizes, phi_layers, q_layers, memory_cap)
        # phi and rho are built in super init call

        # Building q input layer. There is one extra input for behaviour token ID
        self.q_input_layer = tf.keras.layers.Dense(phi_layers[-2] + static_state_size + 1, activation='relu')

        # Building hidden layers
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in q_layers]

        # Building output layer
        self.output_layer = tf.keras.layers.Dense(22, activation='linear')

    def select_action(self, obs, step):
        # get the probabiltiy of selecting a random action instead of e-greedy from policy
        rate = math.exp(-1*step*self.e_decay)

        if rate > random.random():
            # return random action, exploration rate at current step, indicator that action was random
            return random.randrange(22), rate, False
        else:
            # return argmax_a q(s, a), rate at current step, indicator that action was not random
            #return np.argmax(self(np.atleast_2d(np.atleast_2d(obs).astype('float32')))), rate, True
            return np.argmax(self(obs)), rate, True


class ProposedAgent(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False, VIEW_DISTANCE=30):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose, VIEW_DISTANCE)

        # predeclared variables
        self.behaviour_net = None
        self.throttle_net = None

    def flatten_obs(self, env_obs):
        # flatten observations to something passable to tensorflow
        static = np.array([env_obs["number"], env_obs["speed"], env_obs["accel"], env_obs["laneid"], env_obs["lanepos"],
                           env_obs["dti"], env_obs["lanes"][0], env_obs["lanes"][1]])

        vehicles = []

        if "vehicles" in env_obs.keys():
            for vehicle in env_obs["vehicles"]:
                vehicles.append(np.array([vehicle[key] for key in vehicle.keys()]))

        lights = []

        if "lights" in env_obs.keys():
            for light in env_obs["lights"]:
                lights.append(np.array([light[key] for key in light.keys()]))

        # return static vector followed by dynamic array observation vector
        return static, [vehicles, lights]

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
        b_phi_layers = [20, 80]
        t_layers = [200, 100, 100]
        t_phi_layers = [20, 80]

        # start environment
        self.test_env = RoundaboutEnv(self)

        # declare replay buffer experience format
        Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'state_primes', 'terminates'])

        # create policy networks and target networks with equivalent starting parameters
        self.behaviour_net = BehaviourNet(static_input_size=8, variable_input_size=4, phi_layers=b_phi_layers,
                                          q_layers=b_layers, memory_cap=replay_cap)
        self.target_behaviour_net = BehaviourNet(static_input_size=8, variable_input_sizes=[4, 1], phi_layers=b_phi_layers,
                                          q_layers=b_layers, memory_cap=replay_cap)
        policy_params = self.behaviour_net.trainable_variables
        target_params = self.target_behaviour_net.trainable_variables

        for pvar, tvar in zip(policy_params, target_params):
            tvar.assign(pvar.numpy())

        self.throttle_net = BTNet(static_state_size=8, variable_state_sizes=[4, 1], phi_layers=t_phi_layers,
                                          q_layers=t_layers, memory_cap=replay_cap)
        self.target_throttle_net = BTNet(static_state_size=8, variable_state_sizes=[4, 1], phi_layers=t_phi_layers,
                                          q_layers=t_layers, memory_cap=replay_cap)

        policy_params = self.throttle_net.trainable_variables
        target_params = self.throttle_net.trainable_variables

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
            loss_history = []
            terminated = False

            #HER
            states_reached_succesfully = []

            while not terminated:
                obs = self.flatten_obs(self.test_env.sample())
                step += 1

                # select behaviour according to e-greedy policy
                b_action, _, _ = self.behaviour_net.select_action(obs, step)
                # add selected behaviour to observation to pass to throttle net
                t_static = np.append(obs[0], b_action)
                t_dynamic = obs[1]
                t_action, _, _ = self.throttle_net.select_action((t_static, t_dynamic), step)
                state_prime, reward, terminated = self.test_env.step([b_action, t_action])
                episode_return += reward

                # store experience in replay buffer
                self.throttle_net.add_mem(Experience((t_static, t_dynamic), t_action, reward, state_prime, terminated))
                self.behaviour_net.add_mem(Experience(obs, b_action, reward, state_prime, terminated))

                # replay buffer sampling
                if self.throttle_net.mem_counter > batch_size:
                    # collect replay sample
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

                    # calculate behaviour loss and apply grad descent
                    q_prime = np.max(self.target_behaviour_net(np.atleast_2d(behaviour_state_primes).astype('float32')),
                                     axis=1)
                    q_optimal = np.where(behaviour_terminates, behaviour_rewards, behaviour_rewards + gamma * q_prime)
                    q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                    with tf.GradientTape() as tape:
                        q = tf.math.reduce_sum(
                            self.behaviour_net(np.atleast_2d(behaviour_states).astype('float32'))
                            * tf.one_hot(behaviour_actions, 3), axis=1)

                        b_loss = tf.math.reduce_mean(tf.square(q_optimal - q))

                    # update the policy network weights using ADAM
                    b_variables = self.behaviour_net.trainable_variables
                    b_gradients = tape.gradient(b_loss, b_variables)
                    optimizer.apply_gradients(zip(b_gradients, b_variables))

                    # calculate throttle loss and apply grad descent
                    q_prime = np.max(self.target_throttle_net(np.atleast_2d(throttle_state_primes).astype('float32')),
                                     axis=1)
                    q_optimal = np.where(throttle_terminates, throttle_rewards, throttle_rewards + gamma * q_prime)
                    q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                    with tf.GradientTape() as tape:
                        q = tf.math.reduce_sum(
                            self.throttle_net(np.atleast_2d(throttle_states).astype('float32'))
                            * tf.one_hot(throttle_actions, 22), axis=1)

                        t_loss = tf.math.reduce_mean(tf.square(q_optimal - q))

                    # update using ADAM
                    t_variables = self.throttle_net.trainable_variables
                    t_gradients = tape.gradient(t_loss, t_variables)
                    optimizer.apply_gradients(zip(t_gradients, t_variables))








    def select_action(self, time_step):
        # actions may not be selected without training
        if (self.throttle_net is None) or (self.behaviour_net is None):
            raise RuntimeError("[!!] Throttle net and Behaviour net must be trained or loaded before calling select_action.")