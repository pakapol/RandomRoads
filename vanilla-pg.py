# Vanilla Policy Gradient for near-symmetric multi-agent traffic task
#
# Adapted from a code base developed by Sam Greydanus (October 2016)
# https://gist.github.com/greydanus/5036f784eec2036252e1990da21eda18
#
# which, in turn, was adapted from a code base developed by Andrej Karpathy (May 31, 2016)
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
#

from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
from RandomRoads import RandomRoadsEnvironment

class config:
  n_obs = 6 # 6 x 4, properly permuted
  h1 = 48
  h2 = 24
  n_actions = 5
  num_agents = 4
  gamma = .99
  max_grad_norm = 10.
  batch_size = 1024
  max_episode_len = 1000
  batch_size = 5000
  test_every = 50

def discount(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    running_add = running_add * config.gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

class VanillaPGModel(object):

  def __init__(self):
    self.add_placeholder()
    self.setup_reward()
    self.add_prediction_op()
    self.add_optimization_op()
    
  def add_placeholder(self):
    self.x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, config.n_obs * 4])
    self.y_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, config.n_actions])
    self.epr_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
  
  def setup_reward(self):
    reward_mean, reward_var = tf.nn.moments(self.epr_placeholder, [0], shift=None)
    self.advantage = (self.epr_placeholder - reward_mean) / (reward_var + 1e-6)

  def add_prediction_op(self):
    initer = tf.contrib.layers.xavier_initializer()
    W11 = tf.Variable(initer([config.n_obs, config.h1]))
    W12 = tf.Variable(initer([config.n_obs, config.h1]))
    W2 = tf.Variable(initer([config.h1, config.h2]))
    W3 = tf.Variable(initer([config.h2, config.n_actions]))
    b1 = tf.Variable(tf.random_normal([1,config.h1]) * 1e-3)
    b2 = tf.Variable(tf.random_normal([1,config.h2]) * 1e-3)
    b3 = tf.Variable(tf.random_normal([1,config.n_actions]) * 1e-3)
    W1 = tf.concat([W11, W12, W12, W12], axis=0)
    H1 = tf.nn.relu(tf.matmul(self.x_placeholder, W1) + b1)
    H2 = tf.nn.relu(tf.matmul(H1, W2) + b2)
    self.logit_p = tf.nn.softmax(tf.matmul(H2, W3) + b3)
  
  def add_optimization_op(self):
    self.loss = tf.nn.l2_loss(self.y_placeholder - self.logit_p)
    optimizer = tf.train.AdamOptimizer()
    grads = optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(),\
                                        grad_loss=tf.expand_dims(self.advantage, axis=1))
    self.train_op = optimizer.apply_gradients(grads)

  def sample_policy(self, sess, x):
    input_feed = {self.x_placeholder: x}
    aprobs = sess.run(self.logit_p, input_feed)
    actions = np.sum(np.cumsum(aprobs, axis=1) < np.random.rand(config.num_agents,1), axis=1)
    one_hot_actions = np.zeros_like(aprobs)
    one_hot_actions[np.arange(config.num_agents), actions] = 1.
    return actions, one_hot_actions

  def collect_episode_experience(self, sess, env):
    done = np.array([False] * 4)
    obs = env.reset()
    xs, rs, ys, dones = [], [], [], []
    while False in done and len(rs) < config.max_episode_len:
      permuter = [0,1,2,3,1,2,3,0,2,3,0,1,3,0,1,2]
      x = obs[permuter].reshape(4,-1)
      actions, labels = self.sample_policy(sess, x)
      obs, rewards, new_done = env.step(actions)
      xs.append(x)
      rs.append(rewards)
      ys.append(labels)
      dones.append(done)
      done = new_done
    elen = np.sum(np.logical_not(np.array(dones)), axis=0)
    xs, rs, ys = np.array(xs), np.array(rs), np.array(ys)
    x4, r4, y4 = [], [], []
    for i in range(4):
      r4.append(discount(rs[:elen[i],i]))
      x4.append(xs[:elen[i], i, :])
      y4.append(ys[:elen[i], i, :])
    return x4, r4, y4

  def collect_batch_experience(self, sess, env):
    xs, ys, rs = [], [], []
    num_rows = 0
    num_ep = 0
    while num_rows < config.batch_size:
      x, r, y = self.collect_episode_experience(sess, env)
      xs.extend(x)
      ys.extend(y)
      rs.extend(r)
      num_rows += sum(map(len,r))
      num_ep += 1
    return np.vstack(xs), np.hstack(rs), np.vstack(ys), num_ep

  def test_play(self, sess, env):
    _, rs, _ = self.collect_episode_experience(sess, env)
    return np.sum(rs)

  def run(self, sess, env):
    running_reward = None
    num_ep_elapsed = 0
    num_test = 0
    while 1:
      xs, rs, ys, num_ep = self.collect_batch_experience(sess, env)
      reward = np.sum(rs)
      running_reward = reward if running_reward is None else running_reward * 0.99 + reward * 0.01
      input_feed = {self.x_placeholder: xs,
                    self.y_placeholder: ys,
                    self.epr_placeholder: rs}
      _ = sess.run([self.train_op], input_feed)

      num_ep_elapsed += num_ep
      if num_ep_elapsed >= config.test_every * (num_test + 1):
        reward = self.test_play(sess, env)
        print("Episode {}: Reward = {}".format(num_ep_elapsed, reward))
        num_test += 1
        # tf.train.Saver().save(sess, 'model')

def main(_):
  env = RandomRoadsEnvironment()
  m = VanillaPGModel()
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    m.run(sess, env)

if __name__ == '__main__':
  tf.app.run()
