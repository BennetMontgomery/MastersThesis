# IMPORTS
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from agents.agent import Agent
from roundabout_gym.roundabout_env import RoundaboutEnv
from collections import namedtuple
from datetime import datetime
from agents.behaviour_net import BehaviourNet
from copy import deepcopy
from parameters.simulator_params import maximum_npcs
from parameters.hyperparameters import variable_input_size, static_input_size, gamma, alpha, num_heads_b, \
    b_action_space_size, t_action_space_size

# CONSTANTS
SAVE_MODEL = True
MODEL_DIR = "./models"

class ProposedAgent(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False, VIEW_DISTANCE=30):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose, VIEW_DISTANCE)

        self.eval_env = None

        # predeclared variables
        self.behaviour_net = None
        self.throttle_net = None
        self.throttle_path = None
        self.behaviour_path = None

    def save_model(self, model_name):
        os.system(f"mkdir -p {MODEL_DIR}")
        # model.save(f"{MODEL_DIR}/{model_name}.h5")
        os.system(f"mkdir -p '{MODEL_DIR}/{model_name}'")

        self.behaviour_net.save_weights(f"{MODEL_DIR}/{model_name}/behaviour")
        self.throttle_net.save_weights(f"{MODEL_DIR}/{model_name}/throttle")

    def load_model(self, model):
        self.behaviour_net = BehaviourNet(
            static_input_size=static_input_size,
            variable_input_size=variable_input_size,
            attention_heads=num_heads_b,
            q_layers=[256, 128, b_action_space_size],
            pooler_layers=256,
            attention_out_ff=128,
            attention_in_d=256,
            memory_cap=8000,
            e_decay=0.001
        )

        self.throttle_net = BehaviourNet(
            static_input_size=static_input_size+1,
            variable_input_size=variable_input_size,
            attention_heads=8,
            q_layers=[256, 128, t_action_space_size],
            pooler_layers=256,
            attention_out_ff=128,
            attention_in_d=256,
            memory_cap=8000,
            e_decay=0.001
        )

        self.throttle_net.load_weights(f"{MODEL_DIR}/{model}/throttle")
        self.behaviour_net.load_weights(f"{MODEL_DIR}/{model}/behaviour")
        # return tf.keras.models.load_model(f"{MODEL_DIR}/{model}")

    def flatten_obs(self, env_obs):
        # flatten observations to something passable to tensorflow
        obs = [[it+1 for it in [env_obs["number"], env_obs["speed"], env_obs["accel"], env_obs["laneid"],
                           env_obs["lanepos"], env_obs["dti"], env_obs["lanes"][0], env_obs["lanes"][1]]]]

        # populate dynamic observation vector
        if "vehicles" in env_obs.keys():
            for vehicle in env_obs["vehicles"]:
                obs.append([it+1 for it in [vehicle[keya][keyb] for keya in vehicle.keys() for keyb in vehicle[keya]]])

        # pad dynamic observation vector
        for i in range(len(obs[1:]), maximum_npcs):
            obs.append([0 for _ in range(len(obs[0]))])

        for vec in obs:
            for i in range(len(vec), len(obs[0])):
                vec.append(0)

        # return static vector followed by dynamic array observation vector
        return obs

    def train_nets(self, episodes=2000, replay_cap=8000, batch_size=32):
        # UNIVERSAL HYPERPARAMS
        log_freq = 10 # number of rounds between printing of loss and other ML statistics
        optimizer = tf.optimizers.Adam(alpha)
        checkpoint_freq = 100

        # BEHAVIOUR SPECIFIC HYPERPARAMS
        b_q_layers = [256, 128, b_action_space_size] # layer parameters in the behavioural q network
        update_freq_b = 25 # number of rounds between updates to the target q bnet
        encoder_layer_b = 256 # width of the attention embeddings
        encoder_feed_forward_b = 128 # width of the attention post-processing network
        pooler_layers_b = encoder_layer_b # width and depth of the seq-2-vec processing network
        e_decay_b = 0.001 # epsilon decay for random action selection

        # THROTTLE SPECIFIC HYPERPARAMS
        t_action_space_size = 22
        t_q_layers = [256, 128, t_action_space_size]  # layer parameters in the throttle q network
        update_freq_t = 25  # number of rounds between updates to the target q tnet
        num_heads_t = 8  # number of attention heads
        encoder_layer_t = 256  # width of the attention embeddings
        encoder_feed_forward_t = 128  # width of the attention post-processing network
        pooler_layers_t = encoder_layer_b  # width and depth of the seq-2-vec processing network
        e_decay_t = 0.001  # epsilon decay for random action selection

        # start environment
        self.test_env = RoundaboutEnv(self)

        # declare replay buffer memory format
        Experience = namedtuple('Experience', ['states', 'actions', 'reward', 'state_primes', 'terminates'])

        # CREATE NETWORKS
        self.behaviour_net = BehaviourNet(
            static_input_size=static_input_size,
            variable_input_size=variable_input_size,
            attention_heads=num_heads_b,
            q_layers=b_q_layers,
            pooler_layers=pooler_layers_b,
            attention_out_ff=encoder_feed_forward_b,
            attention_in_d=encoder_layer_b,
            memory_cap=replay_cap,
            e_decay=e_decay_b
        )

        self.target_behaviour_net = BehaviourNet(
            static_input_size=static_input_size,
            variable_input_size=variable_input_size,
            attention_heads=num_heads_b,
            q_layers=b_q_layers,
            pooler_layers=pooler_layers_b,
            attention_out_ff=encoder_feed_forward_b,
            attention_in_d=encoder_layer_b,
            memory_cap=replay_cap,
            e_decay=e_decay_b
        )

        self.throttle_net = BehaviourNet(
            static_input_size=static_input_size+1,
            variable_input_size=variable_input_size,
            attention_heads=num_heads_t,
            q_layers=t_q_layers,
            pooler_layers=pooler_layers_t,
            attention_out_ff=encoder_feed_forward_t,
            attention_in_d=encoder_layer_t,
            memory_cap=replay_cap,
            e_decay=e_decay_t
        )

        self.target_throttle_net = BehaviourNet(
            static_input_size=static_input_size+1,
            variable_input_size=variable_input_size,
            attention_heads=num_heads_t,
            q_layers=t_q_layers,
            pooler_layers=pooler_layers_t,
            attention_out_ff=encoder_feed_forward_t,
            attention_in_d=encoder_layer_t,
            memory_cap=replay_cap,
            e_decay=e_decay_t
        )

        print(self.behaviour_net.layers)
        print(self.throttle_net.layers)

        # track reward history
        reward_history = np.empty(episodes)

        total_step = 0

        # train
        for episode in range(episodes):
            print(f"Episode {episode}")

            self.test_env.reset()

            # generic training variables
            episode_return = 0
            step = 0
            b_loss_history = []
            t_loss_history = []
            terminated = False

            while not terminated:
                obs = self.test_env.sample()
                obs = self.flatten_obs(obs)
                if len(obs[0]) > 9:
                    print(obs)
                obstype = type(obs)
                if self.behaviour_net is None:
                    print(f"No network at timestep {step} in episode {episode}")
                    exit(1)

                # build weights if not built
                try:
                    weights_b = self.behaviour_net.trainable_variables
                    weights_b_target = self.target_behaviour_net.trainable_variables
                    weights_t = self.throttle_net.trainable_variables
                    weights_t_target = self.target_throttle_net.trainable_variables
                except ValueError:
                    print("Should've been constructed...")
                    self.behaviour_net(obs)
                    self.target_behaviour_net(obs)
                    t_obs_build = deepcopy(obs)
                    for feature in t_obs_build:
                        feature.append(0)
                    self.throttle_net(t_obs_build)
                    self.target_throttle_net(t_obs_build)

                    weights_b = self.behaviour_net.trainable_variables
                    weights_b_target = self.target_behaviour_net.trainable_variables
                    weights_t = self.throttle_net.trainable_variables
                    weights_t_target = self.target_throttle_net.trainable_variables

                step += 1
                total_step += 1

                # select behaviour according to e-greedy policy
                try:
                    b_action, _, _ = self.behaviour_net.select_action(obs, total_step)
                except tf.errors.InvalidArgumentError:
                    print("Behaviour failure ", obs)
                    exit(1)

                # add selected behaviour to observation to pass to throttle net
                t_obs = deepcopy(obs)
                t_obs[0].append(float(b_action))
                for feature in t_obs[1:]:
                    feature.append(0)

                # select throttle action
                try:
                    t_action, _, _ = self.throttle_net.select_action(t_obs, total_step)
                except tf.errors.InvalidArgumentError:
                    print("Throttle failure ", t_obs)
                    exit(1)

                # apply action to environment
                state_prime, reward, terminated, _ = self.test_env.step([b_action, t_action])
                state_prime = self.flatten_obs(state_prime)
                obs_action_prime = np.argmax(self.behaviour_net(state_prime))
                throttle_state_prime = deepcopy(state_prime)
                throttle_state_prime[0].append(obs_action_prime)
                for vehicle in throttle_state_prime[1:]:
                    vehicle.append(0)

                episode_return += reward

                # store experience in replay buffers
                obstype2 = type(obs)
                if len(obs[0]) >= 10:
                    print(obs)

                self.throttle_net.add_mem(Experience(t_obs, t_action, reward, throttle_state_prime, terminated))
                self.behaviour_net.add_mem(Experience(obs, b_action, reward, state_prime, terminated))

                # replay buffer sampling
                if self.throttle_net.replay_manager.counter > batch_size:
                    behaviour_memories = self.behaviour_net.sample_replay_batch(batch_size)
                    throttle_memories = self.throttle_net.sample_replay_batch(batch_size)
                    behaviour_minibatch = Experience(*zip(*behaviour_memories))
                    throttle_minibatch = Experience(*zip(*throttle_memories))

                    behaviour_states = list(behaviour_minibatch[0])
                    behaviour_actions = np.asarray(behaviour_minibatch[1])
                    behaviour_rewards = np.asarray(behaviour_minibatch[2])
                    behaviour_state_primes = np.asarray(behaviour_minibatch[3])
                    behaviour_terminates = np.asarray(behaviour_minibatch[4])

                    throttle_states = list(throttle_minibatch[0])
                    throttle_actions = np.asarray(throttle_minibatch[1])
                    throttle_rewards = np.asarray(throttle_minibatch[2])
                    throttle_state_primes = np.asarray(throttle_minibatch[3])
                    throttle_terminates = np.asarray(throttle_minibatch[4])

                    # UPDATE BEHAVIOUR NETWORK
                    # calculate loss and apply gradient descent
                    debug = self.target_behaviour_net(behaviour_state_primes, training=True)
                    #q_prime = np.max(self.target_behaviour_net(behaviour_state_primes, training=True), axis=1)
                    q_prime = np.max(debug, axis=1)
                    q_optimal = np.where(behaviour_terminates, behaviour_rewards, behaviour_rewards + gamma*q_prime)
                    q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                    with tf.GradientTape() as tape:
                        # q = tf.math.reduce_sum(
                        #     self.behaviour_net(np.atleast_2d(behaviour_states).astype('float32'), training=True)
                        #     * tf.one_hot(behaviour_actions, b_action_space_size), axis=1)
                        q = tf.math.reduce_sum(self.behaviour_net(behaviour_states, training=True) * tf.one_hot(behaviour_actions, b_action_space_size), axis=1)
                        loss = tf.math.reduce_mean(tf.square(q_optimal - q))

                    # Update using ADAM
                    variables = self.behaviour_net.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables))

                    # record loss
                    b_loss_history.append(loss.numpy())

                    # UPDATE THROTTLE NETWORK
                    # calculate loss and apply gradient descent
                    q_prime = np.max(self.target_throttle_net(throttle_state_primes, training=True), axis=1)
                    q_optimal = np.where(throttle_terminates, throttle_rewards, throttle_rewards + gamma * q_prime)
                    q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                    with tf.GradientTape() as tape:
                        # q = tf.math.reduce_sum(
                        #     self.throttle_net(np.atleast_2d(throttle_states).astype('float32'), training=True)
                        #     * tf.one_hot(throttle_actions, t_action_space_size), axis=1)
                        q = tf.math.reduce_sum(self.throttle_net(throttle_states, training=True) * tf.one_hot(throttle_actions, t_action_space_size), axis=1)
                        loss = tf.math.reduce_mean(tf.square(q_optimal - q))

                    # Update using ADAM
                    variables = self.throttle_net.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables))

                    # record loss
                    t_loss_history.append(loss.numpy())
                else:
                    b_loss_history.append(0)
                    t_loss_history.append(0)

                # update target networks
                if total_step % update_freq_b == 0:
                    # print("updating")
                    behaviour_params = self.behaviour_net.trainable_variables
                    optimal_params = self.target_behaviour_net.trainable_variables

                    for bvar, ovar in zip(behaviour_params, optimal_params):
                        ovar.assign(bvar.numpy())

                if total_step % update_freq_t == 0:
                    # print("updating")
                    throttle_params = self.throttle_net.trainable_variables
                    optimal_params = self.target_throttle_net.trainable_variables

                    for tvar, ovar in zip(throttle_params, optimal_params):
                        ovar.assign(tvar.numpy())

                # new episode if a terminal state is reached
                if terminated:
                    break

            reward_history[episode] = episode_return
            average_reward = reward_history[max(0, episode - 100):(episode+1)].mean()

            if episode % log_freq == 0:
                print(f"Episode: {episode} Episode Reward: {episode_return} P100 Average Reward: {average_reward}")

            if episode % checkpoint_freq == 0:
                self.checkpoint_model(episode, reward_history)

        if SAVE_MODEL:
            time = datetime.now()

            self.folder = f"{time}_final"

            self.save_model(self.folder)

            average_reward = [reward_history[max(0, episode-100):(episode+1)].mean() for episode in range(episodes)]

            plt.plot(reward_history)
            plt.title(f"{datetime.now}")
            plt.savefig(f"{MODEL_DIR}/{time}/{time}_final.png")

            plt.plot(average_reward)
            plt.title(f"{time}_final average over time")
            plt.savefig(f"{MODEL_DIR}/{time}_final/{time}_final average.png")
            plt.close()

        self.test_env.close()

    def validate(self, eval_scenario, eval_folder, npcs, network=None, graphical_mode=False):
        options = [eval_scenario, eval_folder, npcs]

        if network is not None:
            self.load_model(network)

        if self.eval_env is None:
            self.eval_env = RoundaboutEnv(self) if graphical_mode is False else RoundaboutEnv(self, render_mode='sumo-gui')

        curr_state = self.flatten_obs(self.eval_env.reset(options=options))
        episode_return = 0
        terminated = False

        while not terminated:
            b_action = np.argmax(self.behaviour_net(curr_state))
            t_obs = deepcopy(curr_state)
            t_obs[0].append(float(b_action))
            for feature in t_obs[1:]:
                feature.append(0)
            t_action = np.argmax(self.throttle_net(t_obs))

            next_state, reward, terminated, _ = self.eval_env.step([b_action, t_action])

            # record reward
            episode_return += reward

            curr_state = self.flatten_obs(next_state)

        return episode_return

    def checkpoint_model(self, time_step, reward_history):
        time = datetime.now()

        self.folder = f"{time}_{time_step}"

        self.save_model(self.folder)

        average_reward = [reward_history[max(0, episode - 100):(episode + 1)].mean() for episode in range(time_step)]

        plt.plot(reward_history)
        plt.title(f"{datetime.now}")
        plt.savefig(f"{MODEL_DIR}/{time}_{time_step}/{time}_{time_step}.png")

        plt.plot(average_reward)
        plt.title(f"{time}_final average over time")
        plt.savefig(f"{MODEL_DIR}/{time}_{time_step}/{time}_{time_step} average.png")
        plt.close()

    def select_action(self, time_step: list[list[float]]):
        # actions may not be selected without training
        if (self.throttle_net is None) or (self.behaviour_net is None):
            raise RuntimeError("[!!] Throttle net and Behaviour net must be trained or loaded before calling select_action.")

        # call behaviour and throttle networks, collect actions
        b_action = np.argmax(self.behaviour_net(time_step, training=False))
        t_obs = deepcopy(time_step)
        t_obs[0].append(float(b_action))
        t_action = np.argmax(self.throttle_net(time_step, training=False))

        return [b_action, t_action]