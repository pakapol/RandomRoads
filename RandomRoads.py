import sys, os, random
import numpy as np
import matplotlib.pyplot as plt

class config:
  collision_reward = -500.
  success_reward = 500.
  board_size = 48

class RandomRoadsEnvironment(object):

  def __init__(self):
    self.reset()

  def reset(self):
    """
    Initialize the environments
    
    To-do:
      Randomize the targets until all are different
        
      Randomize the junction position of the roads
        road_x : x of the leftmost position of the vertical road
        road_y : y of the top most position of the horizontal road
        Possible range is [6,20] inclusive

      Draw boards components that are invariant
        - Draw black background
        - Draw white roads
        - Draw the four targets
        board : tensor, type float32, [32 x 32 x 4]
     
      Initialize Positions
        x0 = 1   y+1 0  0 (left)
        x1 = x+1 1   0  0 (top)
        x2 = 27  y+1 0  0 (right)
        x3 = x+1 27  0  0 (bottom)

      Initialize the pixels observation
        
    """
    # Random target
    self.target_index = random.choice([[1,0,3,2],[1,2,3,0],[1,3,0,2],
                                  [2,0,3,1],[2,3,0,1],[2,3,1,0],
                                  [3,0,1,2],[3,2,0,1],[3,2,1,0]])
    
    target_assignee = [self.target_index.index(i) for i in range(4)]

    # Random Road x, Road y
    self.road_x = x = np.random.randint(12,config.board_size-5 - 12) #6-53
    self.road_y = y = np.random.randint(12,config.board_size-5 - 12) #6-53
    self.road_ref = np.array([x+1,y+1]).astype(int)
    # Initialize positions and velocities
    self.xstate = np.array([[1, y+1],
                           [x+1, 1],
                           [config.board_size-5,y+1],
                           [x+1,config.board_size-5]]).astype(float)
    self.vstate = np.array([[0,0]]*4)

    self.target_pos = self.xstate[self.target_index]
    # Prepare action to dv mapping
    self.get_dv = np.array([[1.,0.], [0.,1.],[-1.,0.],[0.,-1],[0.,0.]])

    # For rendering purpose, initialize time-constant components of the pixels
    self.board = np.zeros((config.board_size,config.board_size,4))
    self.board[x:x+6, :] = 1.
    self.board[:, y:y+6] = 1.
    self.board[0:6, y:y+6, target_assignee[0]] = 0.75
    self.board[x:x+6, 0:6, target_assignee[1]] = 0.75
    self.board[config.board_size-6:config.board_size, y:y+6, target_assignee[2]] = 0.75
    self.board[x:x+6, config.board_size-6:config.board_size, target_assignee[3]] = 0.75

    # Initialize all agents to not done
    self.isdone = np.array([False]*4)

    return self.get_state()

  def get_state(self):
    return np.concatenate([self.xstate - self.road_ref, self.vstate,
                           self.target_pos - self.road_ref], axis=1)
   

  def _is_off_road(self):
    x, y = self.xstate.T
    return np.logical_and(np.logical_or(x <= self.road_x, self.road_x + 2 <= x),
                          np.logical_or(y <= self.road_y, self.road_y + 2 <= y))

  def _current_frame(self):
    """
      Return a 64 x 64 x 4 frame, each with with following values
        Outside   0.
        Road      1.
        Targets   0.75
        Opponents 0.25
        Self      0.5
    """
    ret = np.array(self.board)
    # Lay over players (0.75 for opponents, 0.5 for self)
    for i in range(4):
      x,y = np.rint(self.xstate[i]).astype(int)
      ret[x:x+4, y:y+4, :] = 0.25
      ret[x:x+4, y:y+4, i] = 0.5
    return ret

  def step(self, actions):
    # actions: shape [4] dtype int32
    
    # case on road, offroad
    dv = self.get_dv[actions]
    rewards = - 1. - (np.sum(dv * self.vstate, axis=1) > 0)
    dv -= (self.vstate / 2) * np.expand_dims(self._is_off_road(), 1)
    new_x = self.xstate + (self.vstate + dv) * (1 - np.expand_dims(self.isdone, 1)) / 2
    new_v = (self.vstate + (1 - np.expand_dims(self.isdone, 1)) * dv) * np.logical_and(0 <= new_x, new_x <= config.board_size - 4)
    new_x = np.minimum(np.ones_like(new_x) * (config.board_size - 4), new_x * (new_x > 0))

    # check collision (6 pairs),
    # if collide, update the two x to the next , v to 0, assign reward, assign done
    
    def collide(x1, new_x1, x2, new_x2):
      xr = x2 - x1
      new_xr = new_x2 - new_x1
      dists = map(lambda x: np.linalg.norm(x, float('inf')),\
                            [new_xr, xr/3 + 2*new_xr/3, 2*xr/3 + new_xr/3])
      return min(dists) < 4.

    def success(xi, new_xi, i):
      dists = map(lambda x: np.linalg.norm(x - self.target_pos[i], float('inf')),\
                            [new_xi, xi/3 + 2*new_xi/3, 2*xi/3 + new_xi/3])
      return min(dists) <= 1.

    collision = [False]*4
    for i in range(4):
      for j in range(i):
        if collide(self.xstate[i], new_x[i], self.xstate[j], new_x[j]):
          collision[i] = True
          collision[j] = True
 
    for i in range(4):
      if collision[i] and not self.isdone[i]:
        rewards[i] += config.collision_reward
    
    if sum(collision) > 0:
      self.isdone = [True] * 4

    # else, check success (Provided they dont collide first)
    # if success, update x accordingly, assign reward, assign done
    
    for i in range(4):
      if success(self.xstate[i], new_x[i], i) and not self.isdone[i]:
        rewards[i] += config.success_reward
        self.isdone[i] = True
    
    self.xstate, self.vstate = new_x, new_v
    return self.get_state(), rewards, self.isdone
    
  def display(self, i):
    plt.imshow(self._current_frame()[:,:,i].T, cmap="gray")
    plt.show()

  def seed(self, seed=None):
    pass

  def close(self):
    pass

if __name__ == '__main__':
  env = RandomRoadsEnvironment()
  while 1:
    print(env.step([0,0,0,0]))
    print(env.step([1,1,1,1]))
    env.display(0)
