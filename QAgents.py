from __future__ import print_function, division, absolute_import
import numpy as np
from collections import defaultdict

class SimpleQAgent(object):
  def __init__(self, nS, nA, gamma, lr, epsilon, decay=0.999995):
    self.Q = defaultdict(lambda: defaultdict(lambda: 0.))
    self.nS = nS
    self.nA = nA
    self.gamma = gamma
    self.lr = lr
    self.epsilon = epsilon
    self.decay = decay
  def get_optimal_action(self, s):
    qs = self.Q[s]
    qsa = [(qs[a],a) for a in range(self.nA)]
    return max(qsa, key=lambda x: (x[0], np.random.rand()))[1]
  
  def get_action(self, s, testing=False):
    if testing:
      return self.get_optimal_action(s)
    #self.epsilon *= self.decay
    return np.random.randint(0, self.nA) if np.random.rand() < self.epsilon\
           else self.get_optimal_action(s)

  def record(self, s, a, r, sp):
    maxqsp = max([self.Q[sp][ap] for ap in range(self.nA)])
    self.Q[s][a] = (1-self.lr)*self.Q[s][a] + self.lr*(r + self.gamma * maxqsp)
