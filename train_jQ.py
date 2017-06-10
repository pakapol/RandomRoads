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
  print_every = 20

def run(env, Q, ep_cap=1000, testing=False):
  done = [False] * 2
  s1, s2 = env.reset()
  num_steps = 0
  while False in done and num_steps < ep_cap:
    a = Q.get_action(tuple(s1), testing=testing)
    a1,a2 = a//5, a%5
    (s1p,s2p),(r1,r2),done = env.step(np.array([a1,a2]))
    Q.record(tuple(s1),a,r1,tuple(s1p))
    s1, s2 = s1p, s2p
    num_steps += 1
  if testing:
    print("Number of steps: {}".format(num_steps))

def main():
  Q = SimpleQAgent(nS=None, nA=config.num_actions ** 2, gamma=config.gamma, lr=config.lr, epsilon=config.epsilon)
  env = CleaningEnvironment()
  ep_count = 0

  while ep_count < config.max_ep:
    run(env, Q)
    if ep_count % config.print_every == config.print_every - 1:
      print("Testing after episode {}".format(ep_count))
      run(env, Q, testing=True)
      
    ep_count += 1
if __name__ == '__main__':
  main()
