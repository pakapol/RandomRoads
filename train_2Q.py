from __future__ import print_function, division, absolute_import

import numpy as np
from QAgents import SimpleQAgent
from cleaningEnvironment import CleaningEnvironment
class config:
  num_actions = 5
  gamma = 0.999
  lr = 0.2
  epsilon = 0.8
  max_ep = 20000
  switch_agent = 20

def run(env, Q1, Q2, a1_learn=False, a2_learn=False, ep_cap=1000):
  done = [False] * 2
  s1, s2 = env.reset()
  num_steps = 0
  testing= not(a1_learn or a2_learn)
  while False in done and num_steps < ep_cap:
    a1 = Q1.get_action(tuple(s1), testing=testing)
    a2 = Q2.get_action(tuple(s2), testing=testing)
    (s1p,s2p),(r1,r2),done = env.step(np.array([a1,a2]))
    if a1_learn:
      Q1.record(tuple(s1),a1,r1,tuple(s1p))
    if a2_learn:
      Q2.record(tuple(s2),a2,r2,tuple(s2p))
    s1, s2 = s1p, s2p
    num_steps += 1
  if testing:
    print("Number of steps: {}".format(num_steps))

def main():
  Q1 = SimpleQAgent(nS=None, nA=config.num_actions, gamma=config.gamma, lr=config.lr, epsilon=config.epsilon)
  Q2 = SimpleQAgent(nS=None, nA=config.num_actions, gamma=config.gamma, lr=config.lr, epsilon=config.epsilon)
  env = CleaningEnvironment()
  ep_count = 0

  while ep_count < config.max_ep:
    a1_learn = (ep_count % (2 * config.switch_agent)) < config.switch_agent
    run(env, Q1, Q2, a1_learn=a1_learn, a2_learn=(not a1_learn))
    if ep_count % config.switch_agent == config.switch_agent - 1:
      print("Testing after episode {}".format(ep_count))
      run(env,Q1,Q2,a1_learn=False,a2_learn=False)
      
    ep_count += 1
if __name__ == '__main__':
  main()
