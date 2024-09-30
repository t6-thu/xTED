import numpy as np
import ipdb

# Load the npz file
data = np.load("/home/chenqm/projects/cross-domain-trajectory-editing-private/env/maze_data/maze_data_24-01-14-01-56-25.npz")

# Access the arrays stored in the npz file
observations = data['observations']
next_observations = data['next_observations']
actions = data['actions']
rewards = data['rewards']
dones = data['dones']
wind_effects = data['wind_effects']
# ipdb.set_trace()

# Print the shape of the arrays (optional)
print("Observations shape:", observations.shape)
print("Next Observations shape:", next_observations.shape)
print("Actions shape:", actions.shape)
print("Rewards shape:", rewards.shape)
print("Dones shape:", dones.shape)
print("Wind Effects shape:", wind_effects.shape)

# Initialize variables to count trajectory lengths
trajectory_lengths = []
current_length = 0

# Iterate through the 'dones' array and count the length of each trajectory
for done in dones:
    current_length += 1
    if done:
        trajectory_lengths.append(current_length)
        current_length = 0

# Print the trajectory lengths
print("Trajectory lengths:", trajectory_lengths)

# Find the shortest trajectory and count how many trajectories have that length
min_traj_length = min(trajectory_lengths)
num_shortest_traj = trajectory_lengths.count(min_traj_length)

print("Shortest trajectory length:", min_traj_length)
print("Number of shortest trajectories:", num_shortest_traj)

# Close the npz file (recommended)
data.close()