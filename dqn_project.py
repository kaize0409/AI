from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm

    TODOs specify the main areas where logic needs to be added.

    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .

    """

    def __init__(self, env):

        self.env = env
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 2
        self.gamma = 0.99
        # If using e-greedy exploration
        self.eps_start = 0.9
        self.eps_end = 0.05
        # self.eps_decay = 1000  # in episodes
        self.eps_decay = (self.eps_start - self.eps_end) / 1000  # in episodes
        # If using a target network
        self.clone_steps = 5000

        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 10000

        print (self.env.observation_space.shape)
        print ([None] + list(self.env.observation_space.shape))

        # define yours training operations here...
        self.observation_input = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
        self.q_values = self.build_model(self.observation_input)

        # define target network
        self.target_observation_input = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
        self.target_q_values = self.build_model(self.target_observation_input, scope="target")

        #  define your update operations here...
        self.action_input = tf.placeholder(tf.int32, shape=[None])
        self.action_input_one_hot = tf.one_hot(self.action_input, env.action_space.n, dtype=tf.float32)
        self.action_q_val = tf.reduce_sum(tf.multiply(self.q_values, self.action_input_one_hot), reduction_indices=1)
        self.target_q_val = tf.placeholder(tf.float32, [None])
        self.q_val_error = self.huber_loss(self.action_q_val, self.target_q_val)
        #self.q_val_error = tf.reduce_mean(tf.squared_difference(self.target_q_val, self.action_q_val))
        self.update_op = tf.train.AdamOptimizer(0.0005).minimize(self.q_val_error)

        self.num_episodes = 0
        self.num_steps = 0

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

        self.summary_writer = tf.summary.FileWriter("summary")

        self.copy_network = self.copyNetwork()

    def huber_loss(self, y_true, y_pred, max_grad=1.):
        """Calculates the huber loss.

        Parameters
        ----------
        y_true: np.array, tf.Tensor
          Target value.
        y_pred: np.array, tf.Tensor
          Predicted value.
        max_grad: float, optional
          Positive floating point value. Represents the maximum possible
          gradient magnitude.

        Returns
        -------
        tf.Tensor
          The huber loss.
        """
        err = tf.abs(y_true - y_pred, name='abs')
        mg = tf.constant(max_grad, name='max_grad')
        lin = mg * (err - .5 * mg)
        quad = .5 * err * err
        return tf.where(err < mg, quad, lin)

    def build_model(self, observation_input, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values

        Currently returns an op that gives all zeros.
        """
        with tf.variable_scope(scope):
            x = layers.fully_connected(observation_input, 64, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, 32, activation_fn=tf.nn.relu)
            q_vals = layers.fully_connected(x, env.action_space.n, activation_fn=None)
            return q_vals

    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement

        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.

        Currently returns a random action.
        """
        epsilon = 0 if evaluation_mode else self.get_epsilon()
        if np.random.random() < epsilon:
            act = env.action_space.sample()
        else:
            act = np.argmax(self.sess.run(self.q_values, feed_dict={self.observation_input: [obs]}))
        return act

    def update(self, obs, act, next_obs, reward, done):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        """

        target = reward if done else reward + self.gamma * np.max(
            self.sess.run(self.target_q_values, feed_dict={self.target_observation_input: [next_obs]})
        )

        self.sess.run(self.update_op, feed_dict={self.observation_input: [obs], self.action_input: [act],
                                                 self.target_q_val: [target]})

    def perceive(self, obs, action, next_obs, reward, done):

        self.replay_memory.push(obs, action, next_obs, reward, done)

        if len(self.replay_memory) > self.replay_memory.capacity:
            self.replay_memory.pop()

        target = []
        if len(self.replay_memory) > self.batch_size:
            minibatch = self.replay_memory.sample(self.batch_size)
            state_batch=[data[0] for data in minibatch]
            action_batch=[data[1] for data in minibatch]
            next_state_batch=[data[2] for data in minibatch]
            reward_batch=[data[3] for data in minibatch]

            for i in range(self.batch_size):
                done = minibatch[i][4]
                if done:
                    target.append(reward_batch[i])
                else:
                    target.append(reward_batch[i] + self.gamma * np.max(
                        self.sess.run(self.target_q_values, feed_dict={self.target_observation_input: [next_state_batch[i]]}
                                      )))

            self.sess.run(self.update_op, feed_dict={self.observation_input: state_batch, self.action_input: action_batch,
                                                 self.target_q_val: target})

    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()
        all_reward = 0

        #self.sess.run(self.copy_network)

        while not done:
            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)


            all_reward += reward
            self.num_steps += 1
            #self.update(obs, action, next_obs, reward, done)
            self.perceive(obs, action, next_obs, reward, done)
            obs = next_obs
            if self.num_steps % self.clone_steps == 0:
                self.sess.run(self.copy_network)

        print(self.num_steps)
        summary = tf.Summary()
        summary.value.add(tag='Reward', simple_value=all_reward)
        self.summary_writer.add_summary(summary, self.num_episodes)
        self.summary_writer.flush()
        self.num_episodes += 1

    def eval(self, save_snapshot=True):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        ep_steps = 0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print ("Evaluation episode: ", total_reward)
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)

    def copyNetwork(self):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="train")
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def get_epsilon(self):
        current_eps = self.eps_start - self.num_episodes * self.eps_decay
        if current_eps < self.eps_end:
            return self.eps_end
        else:
            return current_eps

def train(dqn):
    for i in count(1):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()


def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    #env.seed(41)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
