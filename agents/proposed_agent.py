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
from parameters.simulator_params import maximum_npcs
from parameters.hyperparameters import variable_input_size, static_input_size

# CONSTANTS
SAVE_MODEL = False
MODEL_DIR = "~/Documents/labs/thesis/simulator/models"

class ProposedAgent(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False, VIEW_DISTANCE=30):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose, VIEW_DISTANCE)

        # predeclared variables
        self.behaviour_net = None
        self.throttle_net = None
        self.throttle_path = None
        self.behaviour_path = None

    def save_model(self, model, model_name):
        os.system(f"mkdir -p {MODEL_DIR}")
        model.save(f"{MODEL_DIR}/{model_name}")

    def load_model(self, model):
        return tf.keras.models.load_model(f"{MODEL_DIR}/{model}")

    def flatten_obs(self, env_obs):
        # flatten observations to something passable to tensorflow
        obs = [[-1 if it == 0 else it for it in [env_obs["number"], env_obs["speed"], env_obs["accel"], env_obs["laneid"],
                           env_obs["lanepos"], env_obs["dti"], env_obs["lanes"][0], env_obs["lanes"][1]]]]

        # populate dynamic observation vector
        if "vehicles" in env_obs.keys():
            for vehicle in env_obs["vehicles"]:
                obs.append([-1 if it == 0 else it for it in [vehicle[keya][keyb] for keya in vehicle.keys() for keyb in vehicle[keya]]])

        # pad dynamic observation vector
        for i in range(len(obs[1:]), maximum_npcs):
            obs.append([0 for _ in range(static_input_size)])

        for vec in obs:
            for i in range(len(vec), static_input_size):
                vec.append(0)

        # return static vector followed by dynamic array observation vector
        return obs

    def train_nets(self, episodes=2000, replay_cap=8000, batch_size=32):
        # UNIVERSAL HYPERPARAMS
        gamma = 0.9 # reward discount factor
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
            static_input_size=static_input_size,
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

        # train
        for episode in range(episodes):
            if episode == 1:
                # ensure starting equality of networks
                policy_params = self.behaviour_net.trainable_variables
                target_params = self.target_behaviour_net.trainable_variables

                for pvar, tvar in zip(policy_params, target_params):
                    tvar.assign(pvar.numpy())

                policy_params = self.throttle_net.trainable_variables
                target_params = self.target_throttle_net.trainable_variables

                for pvar, tvar in zip(policy_params, target_params):
                    tvar.assign(pvar.numpy())

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
                step += 1

                # select behaviour according to e-greedy policy
                try:
                    b_action, _, _ = self.behaviour_net.select_action(obs, step)
                except tf.python.framework.errors_impl.InvalidArgumentError:
                    print(obs)
                    exit(1)

                # add selected behaviour to observation to pass to throttle net
                t_obs = obs.copy()
                t_obs[0].append(float(b_action))

                # select throttle action
                t_action, _, _ = self.throttle_net.select_action(obs, step)

                # apply action to environment
                state_prime, reward, terminated, _ = self.test_env.step([b_action, t_action])
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
                    q_prime = np.max(self.target_behaviour_net(behaviour_state_primes, training=True), axis=1)
                    q_optimal = np.where(behaviour_terminates, behaviour_rewards, behaviour_rewards + gamma*q_prime)
                    q_optimal = tf.convert_to_tensor(q_optimal, dtype='float32')
                    with tf.GradientTape() as tape:
                        q = tf.math.reduce_sum(
                            self.behaviour_net(behaviour_states, training=True).astype('float32')
                            * tf.one_hot(behaviour_actions, b_action_space_size), axis=1)
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
                        q = tf.math.reduce_sum(
                            self.throttle_net(throttle_states, training=True).astype('float32')
                            * tf.one_hot(throttle_actions, t_action_space_size), axis=1)
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
                if step % update_freq_b == 0:
                    behaviour_params = self.behaviour_net.trainable_variables
                    optimal_params = self.target_behaviour_net.trainable_variables

                    for bvar, ovar in zip(behaviour_params, optimal_params):
                        ovar.assign(bvar.numpy())

                if step % update_freq_t == 0:
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

        if SAVE_MODEL:
            time = datetime.now()

            self.throttle_folder = f"Throttle_{time}"
            self.behaviour_folder = f"Behaviour_{time}"

            self.save_model(self.throttle_net, self.throttle_folder)
            self.save_model(self.behaviour_net, self.behaviour_folder)

            average_reward = [reward_history[max(0, episode-100):(episode+1)].mean() for episode in range(episodes)]

            plt.plot(reward_history)
            plt.title(f"{datetime.now}")
            plt.savefig(f"{MODEL_DIR}/{time}.png")

            plt.plot(average_reward)
            plt.title(f"{time} average over time")
            plt.savefig(f"{MODEL_DIR}/{time} average.png")
            plt.close()

        self.test_env.close()

    def validate(self, throttle_model=None, behaviour_model=None, write_to_file=True, eval_eps=1):
        throttle = self.load_model(throttle_model) if throttle_model is not None else self.throttle_net
        behaviour = self.load_model(behaviour_model) if behaviour_model is not None else self.behaviour_net

        self.eval_env = RoundaboutEnv(self)
        returns = []
        terminated = False

        for episode in range(eval_eps):
            curr_state = self.eval_env.reset()
            episode_return = 0

            while not terminated:
                b_action = np.argmax(behaviour(curr_state))
                t_obs = curr_state.copy()
                t_obs[0].append(float(b_action))
                t_action = np.argmax(throttle(t_obs))

                # apply action
                next_state, reward, terminated = self.eval_env.step([b_action, t_action])

                # record reward
                episode_return += reward

                if terminated:
                    returns.append(episode_return)
                    break

                curr_state = next_state

        self.eval_env.close()

        if write_to_file:
            f = open(f"{MODEL_DIR}/{datetime.now()}.log", "w")
            f.write(f"Average: {np.sum(returns)/len(returns)}\n")

            for episode in range(len(returns)):
                f.write(f"Validation Episode: {episode} Episode Return: {returns[episode]}\n")

            f.close()

        return np.sum(returns)/len(returns)

    def select_action(self, time_step: list[list[float]]):
        # actions may not be selected without training
        if (self.throttle_net is None) or (self.behaviour_net is None):
            raise RuntimeError("[!!] Throttle net and Behaviour net must be trained or loaded before calling select_action.")

        # call behaviour and throttle networks, collect actions
        b_action = np.argmax(self.behaviour_net(time_step, training=False))
        t_obs = time_step.copy()
        t_obs[0].append(float(b_action))
        t_action = np.argmax(self.throttle_net(time_step, training=False))

        return [b_action, t_action]