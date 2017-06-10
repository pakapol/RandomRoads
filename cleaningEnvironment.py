from __future__ import print_function, division, absolute_import
import numpy as np
import matplotlib.pyplot as plt

class config:
  board_x = 4
  board_y = 6
  num_agents = 2
  prob_appear = 2 / (board_x * board_y * (board_x + board_y))
  size = 1

class CleaningEnvironment(object):

  def __init__(self):
    self.get_dx = np.array([[0,0],[0,1],[1,0],[0,-1],[-1,0]])
    self.reset() 

  def reset(self):
    self.board = np.zeros((config.board_x, config.board_y))
    self.agents = np.array([np.random.randint(0,config.board_x-config.size,size=config.num_agents),
                            np.random.randint(0,config.board_y-config.size,size=config.num_agents)]).T
    self.clean()
    return self.get_state()

  def get_player_state(self, i, ravel=True):
    return np.concatenate([np.array(self.board).astype(int).ravel(), self.agents[range(i,config.num_agents)+range(i)].astype(int).ravel()])

  def get_state(self):
    return [self.get_player_state(i) for i in range(config.num_agents)]

  def clean(self):
    for x,y in self.agents:
      self.board[x:x+config.size,y:y+config.size] = 1

  def dust(self):
    self.board = np.minimum(self.board, (np.random.rand(config.board_x, config.board_y) > config.prob_appear).astype(int))

  def step(self, actions):
    dx = self.get_dx[actions]
    self.agents = np.maximum(0, np.minimum(self.agents + dx, np.array([config.board_x-config.size, config.board_y-config.size])))
    self.dust()
    self.clean()
    rewards = - 1 - (actions > 90)
    done = [np.sum(1 - self.board) == 0] * config.num_agents
    rewards += 0 if False in done else 500
    return self.get_state(), rewards, done

  def get_frame(self):
    graph = np.array(self.board)
    for x,y in self.agents:
      graph[x:x+config.size,y:y+config.size] = 0.5
    return graph.T

  def display(self):
    plt.imshow(self.get_frame(), cmap='gray')
    plt.show()

  def get_pixel_state(self):
    # board x board x 2 x agent
    ret = np.zeros((config.board_x, config.board_y, 2, config.num_agents))
    ret[:,:,0,:] = np.expand_dims(self.board, 2)
    for i in range(config.num_agents):
      x, y = self.agents[i]
      ret[x:x+config.size,y:y+config.size,1,:] = 0.5
      ret[x:x+config.size,y:y+config.size,1,i] = 1
    return ret
 
if __name__ == '__main__':
  env = CleaningEnvironment()
  while 1:
    env.display()
    print(env.step(np.random.randint(0,5,size=config.num_agents)))
    # print(env.get_pixel_state()[:,:,1,1])
