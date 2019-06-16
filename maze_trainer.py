import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import scipy.signal

from network import AC_Network

from maze_game import maze_game

from random import choice, choices, sample, randrange, random
from time import sleep
from time import time
import os

map_len = 5
batch_size = 256
gamma = 0.995
gae_param = 0.95
num_workers = 256


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


trajectories = []
epsilon = 0.01

lstm_len = 128


class DataHolder():

    def __init__(self, start_rnn_state):
        self.start_rnn_state = start_rnn_state
        self.rnn_state = start_rnn_state
        self.episode_buffer = []
        self.episode_values = []
        self.episode_rewards = []
        self.episode_reward_estimates = []
        self.episode_rnn_states = []
        self.episode_step_count = 0
        self.episode_reward = 0
        self.episode_reward_estimate = 0

    def get_rnn_state(self):
        return self.rnn_state

    def step(self, obs, action, probability, reward, reward_estimate, value, new_rnn_state):
        self.episode_buffer.append([obs, [action], probability])
        self.episode_values.append(value)
        self.episode_rewards.append(reward)
        self.episode_reward_estimates.append(reward_estimate)
        self.episode_rnn_states.append(self.rnn_state)
        self.rnn_state = new_rnn_state
        self.episode_step_count += 1
        self.episode_reward += reward
        self.episode_reward_estimate += reward_estimate

    def end_episode(self):
        self.rewards_plus = np.asarray(self.episode_rewards + [0.0])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(self.episode_values + [0.0])
        advantages = self.episode_rewards + gamma * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma * gae_param)

        mask = [1] * len(self.episode_rewards)
        '''
        indices = []
        non_zero = 0
        for i in range(len(self.episode_rewards)):
            if self.episode_rewards[i] == 0:
                indices.append(i)
            else:
                mask[i] = 1
                non_zero+=1
        '''
        #x = sample(indices,non_zero*10)

        # for i in x:
        #mask[i] = 1

        for i in range(len(self.episode_buffer)):
            self.episode_buffer[i].append(advantages[i])
            self.episode_buffer[i].append(discounted_rewards[i])
            self.episode_buffer[i].append(self.episode_rewards[i])
            self.episode_buffer[i].append(mask[i])

        last_episode_lenght = len(self.episode_buffer) % lstm_len
        dummy_state = self.episode_buffer[-1]
        dummy_state[0] = np.zeros([map_len * map_len + 4])
        dummy_state[3] = 0.0
        dummy_state[4] = 0.0
        dummy_state[6] = 0
        for i in range((lstm_len - last_episode_lenght) % lstm_len):
            self.episode_buffer.append(dummy_state)

        for i in range(0, len(self.episode_buffer), lstm_len):
            if i == len(self.episode_buffer) - lstm_len:
                trajectories.append(
                    (self.episode_rnn_states[i], last_episode_lenght, self.episode_buffer[i:]))
            else:
                trajectories.append(
                    (self.episode_rnn_states[i], lstm_len, self.episode_buffer[i:i + lstm_len]))

        # trajectories.append(episode_buffer)

        return self.episode_step_count, self.episode_reward, self.episode_reward_estimate

    def reset(self):
        self.rnn_state = self.start_rnn_state
        self.episode_buffer = []
        self.episode_values = []
        self.episode_rewards = []
        self.episode_rnn_states = []
        self.episode_reward_estimates = []
        self.episode_step_count = 0
        self.episode_reward = 0
        self.episode_reward_estimate = 0


class TrajectoryGenerator():

    def __init__(self, network, env_number):
        self.network = network
        self.envs = []
        self.data_holders = []
        self.last_obs = []
        self.env_number = env_number
        self.rewards = []
        self.estimates = []
        self.lenghts = []
        self.first_episode_reward = []
        self.second_episode_reward = []
        self.tenth_episode_reward = []
        self.env_steps = 0
        self.summary_writer = tf.summary.FileWriter("train_worker_new")
        for i in range(env_number):
            self.envs.append(maze_game(10, map_len, 10, 300))
            self.data_holders.append(DataHolder(self.network.state_init))
            obs, reward, done = self.envs[-1].reset()
            self.last_obs.append(obs)

    def prepare_lstm_state(self):
        lstm_c = []
        lstm_h = []

        for x in self.data_holders:
            c, h = x.get_rnn_state()
            lstm_c.append(c)
            lstm_h.append(h)

        lstm_c = np.concatenate(lstm_c, 0)
        lstm_h = np.concatenate(lstm_h, 0)

        return lstm_c, lstm_h

    def generate_trajectories(self, lenght):

        preparing_lstm = 0
        calculating_network = 0
        sampling = 0
        doing_step = 0
        updating_stuff = 0

        for _ in range(lenght):

            old = time()

            lstm_c, lstm_h = self.prepare_lstm_state()

            new_t = time()
            preparing_lstm += (new_t - old)
            old = new_t
            # print(lstm_c)
            a_dist, rnn_state, value, sampled, reward_estimate = sess.run([self.network.policy, self.network.state_out, self.network.value, self.network.result, self.network.reward],
                                                                          feed_dict={self.network.inputs: self.last_obs,
                                                                                     self.network.state_in[0]: lstm_c,
                                                                                     self.network.state_in[1]: lstm_h,
                                                                                     self.network.batchsize: [self.env_number],
                                                                                     self.network.sequence_lengths: ([1] * self.env_number)})

            lstm_c, lstm_h = rnn_state

            new_t = time()
            calculating_network += (new_t - old)
            old = new_t
            for i in range(self.env_number):

                old = time()

                # choices([0,1,2,3],weights=a_dist[i])[0]
                action = sampled[i][0]

                new_t = time()
                sampling += (new_t - old)
                old = new_t

                probability = a_dist[i][action]

                new_obs, reward, done, statistics = self.envs[i].step(action)

                new_t = time()
                doing_step += (new_t - old)
                old = new_t
                # print(done)
                val = value[i, 0]
                rew_est = reward_estimate[i, 0]
                local_lstm_state = ([lstm_c[i]], [lstm_h[i]])

                self.data_holders[i].step(
                    self.last_obs[i], action, probability, reward, rew_est, val, local_lstm_state)

                if done:
                    episode_lenght, episode_reward, episode_estimate = self.data_holders[i].end_episode(
                    )
                    self.rewards.append(episode_reward)
                    self.lenghts.append(episode_lenght)
                    self.estimates.append(episode_estimate)
                    self.first_episode_reward.append(statistics[0])
                    self.second_episode_reward.append(statistics[1])
                    self.tenth_episode_reward.append(statistics[9])
                    self.data_holders[i].reset()
                    obs, reward, done = self.envs[i].reset()
                    self.last_obs[i] = obs
                else:
                    self.last_obs[i] = new_obs

                new_t = time()
                updating_stuff += (new_t - old)
                old = new_t

        print("preparing lstm: " + str(preparing_lstm))
        print("calculating network: " + str(calculating_network))
        print("sampling: " + str(sampling))
        print("doing step: " + str(doing_step))
        print("updating stuff: " + str(updating_stuff))
        self.env_steps += lenght * self.env_number

        summary = tf.Summary()
        summary.value.add(tag='Perf/Reward',
                          simple_value=float(np.mean(self.rewards)))
        summary.value.add(tag='Perf/Length',
                          simple_value=float(np.mean(self.lenghts)))
        summary.value.add(tag='Perf/Reward Estimate',
                          simple_value=float(np.mean(self.estimates)))
        summary.value.add(tag='Perf/1RunReward',
                          simple_value=float(np.mean(self.first_episode_reward)))
        summary.value.add(
            tag='Perf/2RunReward', simple_value=float(np.mean(self.second_episode_reward)))
        summary.value.add(tag='Perf/10RunReward',
                          simple_value=float(np.mean(self.tenth_episode_reward)))

        self.summary_writer.add_summary(summary, self.env_steps)

        self.rewards = []
        self.lenghts = []
        self.first_episode_reward = []
        self.second_episode_reward = []
        self.tenth_episode_reward = []
        self.estimates = []
        self.summary_writer.flush()


load_model = False
model_path = './model2'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/gpu:0"):
    global_episodes = tf.Variable(
        0, dtype=tf.int64, name='global_episodes', trainable=False)
    increment = global_episodes.assign_add(1)

    #learning_rate = tf.train.polynomial_decay(1.5e-04, global_episodes,5000, 1e-07,power=0.5)

    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    # Generate global network
    master_network = AC_Network('global', trainer, batch_size, map_len)
    # multiprocessing.cpu_count() # Set workers to number of available CPU threads
    worker = TrajectoryGenerator(master_network, num_workers)
    saver = tf.train.Saver(max_to_keep=5)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

epochs = 4

old_trajectories = []

# most probably useless retraining on the same data after diverged too far -> bigger batches


def train(coordinator):
    train_cycles = 0
    summary_writer = tf.summary.FileWriter("train_trainer_new")
    global trajectories
    global old_trajectories
    while not coord.should_stop():

        old = time()
        worker.generate_trajectories(3000)
        print(len(trajectories))
        new_t = time()
        print("generate: " + str(new_t - old))
        #trajectories = []
        # input()
        if len(trajectories) >= batch_size:
            train_cycles += 1
            # to może się rozwalać z wiadomych powodów
            '''
            advs = []
            for t in trajectories:
                for x in t[2]:
                    advs.append(x[3])

            advs = np.asarray(advs)
            '''
            adv_mean = 0  # advs.mean()
            adv_std = 1  # max(advs.std(),1e-4) * 10

            #print("advantages mean: "+str(adv_mean)+" stddev: "+str(adv_std))
            old = time()
            total_trajectores = old_trajectories + trajectories
            for _ in range(epochs):

                idx = np.arange(len(total_trajectores))
                np.random.shuffle(idx)

                for i in range(0, len(total_trajectores) - batch_size, batch_size):
                    # print(idx[i:i+batch_size])
                    batch = [total_trajectores[x]
                             for x in idx[i:i + batch_size]]

                    rnn_states = [x[0] for x in batch]
                    seq = [x[1] for x in batch]
                    batch = [x[2] for x in batch]

                    observations = []
                    actions = []
                    probabilities = []
                    advantages = []
                    values = []
                    rewards = []
                    masks = []

                    batch_rnn_state = (np.concatenate(
                        [x[0] for x in rnn_states], 0), np.concatenate([x[1] for x in rnn_states], 0))

                    # [obs,joint_action,probability,old_rnn_state]
                    for rollout in batch:
                        for x in rollout:
                            observations.append(x[0])
                            actions.append(x[1])
                            probabilities.append(x[2])
                            advantages.append((x[3] - adv_mean) / adv_std)
                            values.append(x[4])
                            rewards.append(x[5])
                            masks.append(x[6])

                    #sequence_lengths = [x-lstm_len for x in sequence_lengths]

                    feed_dict = {master_network.target_v: values,
                                 master_network.inputs: observations,
                                 master_network.actions: actions,
                                 master_network.old_probabilities: probabilities,
                                 master_network.advantages: advantages,
                                 master_network.target_r: rewards,
                                 master_network.masks: masks,
                                 master_network.state_in[0]: batch_rnn_state[0],
                                 master_network.state_in[1]: batch_rnn_state[1],
                                 master_network.batchsize: [batch_size],
                                 master_network.sequence_lengths: seq}

                    v_l, p_l, e_l, g_n, v_n, r_l, new_batch_rnn_state, _, ratios = sess.run([master_network.value_loss,
                                                                                             master_network.policy_loss,
                                                                                             master_network.entropy,
                                                                                             master_network.grad_norms,
                                                                                             master_network.var_norms,
                                                                                             master_network.reward_loss,
                                                                                             master_network.state_out,
                                                                                             master_network.apply_grads,
                                                                                             master_network.ratios],
                                                                                            feed_dict=feed_dict)

                    #print("min: "+str(min(ratios))+" max: "+str(max(ratios)))

            new_t = time()
            print("train: " + str(new_t - old))
            old = new_t
            sess.run(increment)
            if train_cycles % 5 == 0:

                summary = tf.Summary()
                summary.value.add(tag='Losses/Value Loss',
                                  simple_value=float(v_l))
                summary.value.add(tag='Losses/Policy Loss',
                                  simple_value=float(p_l))
                summary.value.add(tag='Losses/Entropy',
                                  simple_value=float(e_l))
                summary.value.add(tag='Losses/Grad Norm',
                                  simple_value=float(g_n))
                summary.value.add(tag='Losses/Var Norm',
                                  simple_value=float(v_n))
                summary.value.add(tag='Losses/Reward Loss',
                                  simple_value=float(r_l))
                summary_writer.add_summary(summary, train_cycles)

                saver.save(sess, model_path + "/model.ckpt",
                           global_step=global_episodes)
                print("impressive")

            old_trajectories = trajectories
            trajectories = []


with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()

    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    train(coord)
