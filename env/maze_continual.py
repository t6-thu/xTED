import numpy as np
import gym
import datetime
import matplotlib.pyplot as plt
from typing import Tuple


class Robot(object):

    def __init__(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos

    def set_pos(self, obs):
        self.x_pos = obs[0]
        self.y_pos = obs[1]

    def step(self, action):
        delta_x, delta_y = action
        self.x_pos += delta_x
        self.y_pos += delta_y

    def get_pos(self):
        return np.array([self.x_pos, self.y_pos])


class Maze(gym.Env):

    def __init__(self, size: int = 5, wind_pos: int = 0, prob: float = 0.5, record_data: bool = False):
        self.size = size
        self.wind_pos = wind_pos
        self.robot = Robot(0, 0)
        self.init = np.array([0, 0])
        self.target = np.array([size - 1, size - 1])
        self.prob = prob
        self.maze = np.zeros((size, size))
        self.observation_space = np.array([[1],[1]])
        self.action_space = np.array([1])

        self.maze[0:5, 2] = 0.5  # Set the windy cells to 0.5
        self.reset()
        self.record_data = record_data

    def reset(self):
        self._state = self.init
        self.robot.set_pos(self._state)
        self.total_steps = 0
        self.cost = np.sum(self.target - self.init)

        return self._state

    def get_dataset(self):
        
        data = np.load("/home/chenqm/projects/cross-domain-trajectory-editing-private/env/maze_data/maze_data_0.2_24-01-16-22-08-33.npz")
        import ipdb
        # ipdb.set_trace()
        return data

    def _get_next_state(self, action, action_scale=0.5):
        angle = action
        delta_x = np.cos(angle) * action_scale
        delta_y = np.sin(angle) * action_scale

        self.robot.x_pos += delta_x
        self.robot.y_pos += delta_y

        # Check boundaries and adjust the robot's position accordingly
        self.robot.x_pos = max(np.array([0]), min(self.robot.x_pos, np.array([self.size - 1])))
        self.robot.y_pos = max(np.array([0]), min(self.robot.y_pos, np.array([self.size - 1])))

        # Apply wind effect after checking boundaries
        wind_effect = False
        step_in_wind = False
        if 2 <= self.robot.x_pos <= 3 and 0 <= self.robot.y_pos <= 5:
            step_in_wind = True
            if np.random.rand() < self.prob:
                self.robot.x_pos -= 1
                wind_effect = True
        try:
            next_state = np.array([self.robot.x_pos, self.robot.y_pos])
        except:
            import ipdb
            ipdb.set_trace()
        self.total_steps += 1

        return next_state, wind_effect, step_in_wind

    def _in_wind(self, x, y):
        if 2 <= x <= 3 and 0 <= y <= 5:
            return True
        return False

    def step(self, action):
        # prev_pos = self.robot.get_pos()
        next_state, wind_effect, _ = self._get_next_state(action)

        # wind_effect = False
        # # Check if the robot is in the windy zone and if the wind had an effect on its position
        # if 2 <= prev_pos[0] <= 3 and 0 <= prev_pos[1] <= 5:
        #     if self.robot.x_pos == 1:  # Blown to left
        #         wind_effect = True

        reward = self._get_reward()
        done = self._is_done()
        if not self.record_data:  # 仅在record_data为False时调用render方法
            self.render(wind_effect)
        return next_state, reward, done, {'wind_effect': wind_effect}

    def _get_reward(self):
        if np.linalg.norm(self.robot.get_pos() - self.target) <= 0.5:
            return 10  # target
        else:
            return -1  # every timestep -1

    def _is_done(self):
        return np.linalg.norm(self.robot.get_pos() - self.target) <= 1

    def _get_observation(self):
        return np.array([self.robot.x_pos, self.robot.y_pos])
    
    def render(self, wind_effect=False, save = False, i=0):
        plt.cla()
        plt.imshow(self.maze, cmap='gray', origin='lower', alpha=0.5, extent=(0, self.size, 0, self.size))
        plt.scatter(self.target[0] + 0.5, self.target[1] + 0.5, c='red', marker='X', label='Target', s=200)
        plt.scatter(self.robot.x_pos + 0.5, self.robot.y_pos + 0.5, c='blue', marker='o', label='Robot', s=200)
        
        if wind_effect:
            plt.arrow(self.robot.x_pos + 0.5, self.robot.y_pos + 0.5, -1, 0, width=0.1, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
            plt.title('Robot in Maze (Wind Effect)')
        else:
            plt.title('Robot in Maze')
        
        plt.legend()
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.xticks(range(self.size + 1))
        plt.yticks(range(self.size + 1))
        plt.grid()
        plt.pause(0.5)
        if save:
            plt.savefig(f'/home/chenqm/projects/cross-domain-trajectory-editing-private/renders/{i}.png')
        

def expert_policy(robot_pos, target_pos, noise_prob=0.1):
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]

    if np.random.rand() < noise_prob:
        return np.random.uniform(0, 2 * np.pi)

    angle = np.arctan2(dy, dx)
    return angle

if __name__ == "__main__":
    record_data = True  # 设置为True以记录数据
    prob = 0.2
    env = Maze(record_data=record_data, prob=prob)  # 将record_data传递给Maze构造函数

    num_iterations = 1000
    all_observations = []
    all_next_observations = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_wind_effects = []

    for i in range(num_iterations):
        obs = env.reset()
        done = False

        while not done:
            action = expert_policy(env.robot.get_pos(), env.target)
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            wind_effect = info['wind_effect']

            all_observations.append(prev_obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            all_wind_effects.append(wind_effect)

            # if not done:
            all_next_observations.append(obs)

            print(f"Iteration: {i + 1}, Observation: {prev_obs}, Action: {action}, Next_Observation: {obs}, Reward: {reward}, Done: {done}, Wind_effect: {wind_effect}")

    if record_data:
        # 将数据保存为NumPy压缩文件
        nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        np.savez_compressed("maze_data/maze_data_{}_{}.npz".format(prob, nowTime), observations=all_observations, next_observations=all_next_observations, actions=all_actions, rewards=all_rewards, dones=all_dones, wind_effects=all_wind_effects)