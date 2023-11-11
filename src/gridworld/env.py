from enum import IntEnum

from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
from gridworld._gridworld_madrona import GridWorldSimulator, madrona
import gymnasium as gym
from gymnasium import spaces
import sys 
import numpy as np
import torch

num_worlds = 1
map_shape = (32,32)

__all__ = ['GridWorld']

class MapTiles(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2
    TRG = 3

class GridWorldEnv(gym.Env):

    def __init__(self, num_worlds, max_num_agents, map_shape):
        """TODO: Add docstring here"""

        self.max_n_agents = max_num_agents # number of agents
        self.shape = np.array(map_shape) # size of the board 
        self.size = np.prod(self.shape) # size of the board (flattened)
        self.n_worlds = num_worlds # number of worlds

        # Create simulator  
        self.sim = GridWorldSimulator(
                # walls=walls,
                # rewards=rewards,
                # end_cells=end_cells,
                # start_x = start_cell[1],
                # start_y = start_cell[0],
                max_episode_length = 0, # No max
                exec_mode = madrona.ExecMode.CPU,
                num_worlds = num_worlds,
                gpu_id = 0,
            )

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 1, shape=(len(MapTiles),) + tuple(self.shape), dtype=int),
                # "agent_locs": spaces.Box(0, max(self.shape) - 1, shape=(self.max_n_agents, 2), dtype=int),
                # "agent_goals": spaces.Box(0, max(self.shape) - 1, shape=(self.max_n_agents, 2), dtype=int),
                "vector": spaces.Box(0, max(self.shape) - 1, shape=(self.max_n_agents * 4,), dtype=int),
            }
        )
        # Define action space
        # We have 4 actions per agent (F, RL, RR, W)
        self.action_space = spaces.MultiDiscrete((self.max_n_agents, 4))

        self.force_reset = self.sim.reset_tensor().to_torch()
        self.actions = self.sim.action_tensor().to_torch()
        self.observations = self.sim.observation_tensor().to_torch()
        self.rewards = self.sim.reward_tensor().to_torch()
        self.dones = self.sim.done_tensor().to_torch()
        self.map = self.sim.map_tensor().to_torch()

    def get_obs(self):
        assert self.map.shape[0] == self.n_worlds
        im = self.map.reshape(self.n_worlds, *self.shape)
        im = torch.eye(len(MapTiles))[im]
        im = rearrange(im, 'b h w c -> b c h w')
        vec = torch.zeros(self.n_worlds, self.max_n_agents * 4)
        breakpoint()
        return {
            'image': im,
            'vector': vec,
        }

    def step(self, actions: torch.Tensor):
        self.sim.actions = actions
        self.sim.step()
        obs = self.get_obs()
        reward = 0
        done = False
        info = {}
        return 

    def reset(self, seed):
        self.sim.reset_tensor()
        obs = self.get_obs()
        return obs



if __name__ == "__main__":

    env = GridWorld(num_worlds, 2, start_cell, end_cell, rewards, walls)
    breakpoint()
    env.step()

