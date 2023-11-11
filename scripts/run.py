import sys 
import numpy as np
import torch
from gridworld import GridWorld

num_worlds = int(sys.argv[1])

grid_world = GridWorld(num_worlds)
#grid_world.vis_world()

print(grid_world.map)
print(grid_world.map.shape)

print(grid_world.tasks)
print(grid_world.tasks.shape)

print(grid_world.observations.shape)

for i in range(5):
    print("Obs:")
    print(grid_world.observations)

    # "Policy"
    grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    print(grid_world.actions[:,0])
    # grid_world.actions[:, 0] = [1,]
    #grid_world.actions[:, 0] = 3 # right to win given (4, 4) start

    print("Actions:")
    print(grid_world.actions)

    # Advance simulation across all worlds
    grid_world.step()
    
    print("Rewards: ")
    print(grid_world.rewards)
    print("Dones:   ")
    print(grid_world.dones)
    print()
