import numpy as np
import matplotlib.pyplot as plt
from ._gridworld_madrona import GridWorldSimulator, madrona

__all__ = ['GridWorld']

class GridWorld:
    def __init__(self, num_worlds):
        self.sim = GridWorldSimulator(
                max_episode_length = 0, # No max
                exec_mode = madrona.ExecMode.CPU,
                num_worlds = num_worlds,
                gpu_id = 0,
            )

        self.force_reset = self.sim.reset_tensor().to_torch()
        self.actions = self.sim.action_tensor().to_torch()
        self.observations = self.sim.observation_tensor().to_torch()
        self.rewards = self.sim.reward_tensor().to_torch()
        self.dones = self.sim.done_tensor().to_torch()
        self.map = self.sim.map_tensor().to_torch()
        self.tasks = self.sim.task_tensor().to_torch()

    def step(self):
        self.sim.step()

    def jax(self):
        return self.sim.jax(True)
