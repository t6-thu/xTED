'''
  Replay buffer for storing samples.
  @python version : 3.6.8
'''
import torch
import d4rl
import numpy as np
from gym.spaces import Box, Discrete, Tuple

# from envs import get_dim


class Buffer(object):

    def append(self, *args):
        pass

    def sample(self, *args):
        pass

class ReplayBuffer(Buffer):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = torch.device(device)

    def add_sample(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return {
            'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
            'actions': torch.FloatTensor(self.action[ind]).to(self.device), 
            'rewards': torch.FloatTensor(self.reward[ind]).to(self.device), 
            'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device), 
            'dones': torch.FloatTensor(self.done[ind]).to(self.device)
            }