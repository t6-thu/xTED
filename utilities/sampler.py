# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent trajectory samplers."""

import time
from typing import Callable

import numpy as np
import torch
from env import get_envs
import ipdb

WIDTH = 250
HEIGHT = 200


class StepSampler(object):
    def __init__(self,
        env_fn: Callable,
        num_envs: int = 1,
        seed: int = 0,
        max_traj_length: int = 1000):
        
        self.max_traj_length = max_traj_length
        # self._env = env
        self._traj_steps = 0
        
        
        self._env = env_fn()
        self._envs = get_envs(env_fn, num_envs)
        self._envs.seed(seed)
        self._num_envs = num_envs
        
        

        
        # ipdb.set_trace()

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        ready_env_ids = np.arange(self._num_envs)
        self._current_observation = self._envs.reset(ready_env_ids)[0]
        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                observation.reshape(1, -1), deterministic=deterministic
            ).reshape(-1)
            next_observation, reward, done, _ = self._envs.step(action, ready_env_ids)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self._envs.reset()

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env

class MazeSampler(object):
    def __init__(
        self,
        env,
        max_traj_length: int = 1000,
    ):
        self.env = env
        self.max_traj_length = max_traj_length
        
    def set_normalizer(self, normalizer):
        self._normalizer = normalizer
        
    def sample(self, policy, num_trajs, deterministic):
        trajs = []
        i = 0
        for _ in range(num_trajs):
            traj = []
            state = self.env.reset()  # Replace with your initial state
            done = False
            print(i)
            i += 1
            while not done:
                # Use your policy to get an action based on the current state
                # import ipdb
                # ipdb.set_trace()
                action = policy(observations = state, deterministic = deterministic)

                # Simulate the environment and get the next state, reward, and done flag
                next_state, reward, done, info = self.env.step(action)  # Replace with your environment simulation function

                # Append the tuple (state, action, reward, next_state) to the trajectory
                traj.append((state, action, reward, next_state, done, info))

                # Update the current state for the next iteration
                state = next_state

            # Append the trajectory to the list of trajectories
            trajs.append(traj)

        return trajs

class TrajSampler(object): 
    # eval_sampler
    
    def __init__(
        self,
        env_fn: Callable,
        num_envs: int,
        seed: int,
        max_traj_length: int = 1000,
        render: bool = False,
    ):
        self.max_traj_length = max_traj_length
        self._env = env_fn()
        self._envs = get_envs(env_fn, num_envs)
        self._envs.seed(seed)
        self._num_envs = num_envs
        self._render = render
        self._normalizer = None

    def set_normalizer(self, normalizer):
        self._normalizer = normalizer

    def sample(
        self,
        policy,
        n_trajs: int,
        deterministic: bool = False,
        env_render_fn: str = "render",
    ):
        assert n_trajs > 0
        ready_env_ids = np.arange(min(self._num_envs, n_trajs))

        observation, _ = self.envs.reset(ready_env_ids)
        # import ipdb
        # ipdb.set_trace()
        observation = self._normalizer.normalize(observation, "observations")

        observations = [[] for i in range(len(ready_env_ids))]
        actions = [[] for _ in range(len(ready_env_ids))]
        rewards = [[] for _ in range(len(ready_env_ids))]
        next_observations = [[] for _ in range(len(ready_env_ids))]
        dones = [[] for _ in range(len(ready_env_ids))]

        trajs = []
        n_finished_trajs = 0
        i = 0
        while True:
            action = policy(observation, deterministic=deterministic)
            # import ipdb
            # ipdb.set_trace()
            action = self._normalizer.unnormalize(action, "actions")
            next_observation, reward, terminated, truncated, _ = self.envs.step(
                action, ready_env_ids
            )
            # ipdb.set_trace()
            done = np.logical_or(terminated, truncated)
            if self._render:
                getattr(self.envs, env_render_fn)()
                time.sleep(0.01)
            # ipdb.set_trace()
            next_observation = self._normalizer.normalize(
                next_observation, "observations"
            )
            
            for idx, env_id in enumerate(ready_env_ids):
                observations[env_id].append(observation[idx])
                actions[env_id].append(action[idx])
                rewards[env_id].append(reward[idx])
                next_observations[env_id].append(next_observation[idx])
                dones[env_id].append(done[idx])

            if np.any(done):
                # import ipdb
                # ipdb.set_trace()
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                # print(i)
                i += 1
                for ind in env_ind_local:
                    trajs.append(
                        dict(
                            observations=np.array(observations[ind], dtype=np.float32),
                            actions=np.array(actions[ind], dtype=np.float32),
                            rewards=np.array(rewards[ind], dtype=np.float32),
                            next_observations=np.array(
                                next_observations[ind], dtype=np.float32
                            ),
                            dones=np.array(dones[ind], dtype=np.float32),
                        )
                    )
                    observations[ind] = []
                    actions[ind] = []
                    rewards[ind] = []
                    next_observations[ind] = []
                    dones[ind] = []

                n_finished_trajs += len(env_ind_local)
                if n_finished_trajs >= n_trajs:
                    trajs = trajs[:n_trajs]
                    break

                # surplus_env_num = len(ready_env_ids) - (n_trajs - n_finished_trajs)
                # if surplus_env_num > 0:
                #     mask = np.ones_like(ready_env_ids, dtype=bool)
                #     mask[env_ind_local[:surplus_env_num]] = False
                #     ready_env_ids = ready_env_ids[mask]

                obs_reset, _ = self.envs.reset(env_ind_global)
                obs_reset = self._normalizer.normalize(obs_reset, "observations")
                next_observation[env_ind_local] = obs_reset

            observation = next_observation
        return trajs

    @property
    def env(self):
        return self._env

    @property
    def envs(self):
        return self._envs
    
    
class OnlineTrajSampler(object): 
    # eval_sampler
    
    def __init__(
        self,
        env_fn: Callable,
        num_envs: int,
        seed: int,
        max_traj_length: int = 1000,
        render: bool = False,
    ):
        self.max_traj_length = max_traj_length
        self._env = env_fn()
        self._envs = get_envs(env_fn, num_envs)
        self._envs.seed(seed)
        self._num_envs = num_envs
        self._render = render
        self._normalizer = None

    def set_normalizer(self, normalizer):
        self._normalizer = normalizer

    def sample(
        self,
        policy,
        n_trajs: int,
        deterministic: bool = False,
        env_render_fn: str = "render",
        joint_noise_mean: float = 0.0,
        joint_noise_std: float = 1.0,
    ):
        policy.eval()
        assert n_trajs > 0
        # import ipdb
        # ipdb.set_trace()
        transition_num = n_trajs * self.max_traj_length
        ready_env_ids = np.arange(min(self._num_envs, n_trajs))

        observation, _ = self.envs.reset(ready_env_ids)
        # observation = self._normalizer.normalize(observation, "observations")

        observations = [[] for i in range(len(ready_env_ids))]
        actions = [[] for _ in range(len(ready_env_ids))]
        rewards = [[] for _ in range(len(ready_env_ids))]
        next_observations = [[] for _ in range(len(ready_env_ids))]
        dones = [[] for _ in range(len(ready_env_ids))]

        trajs = []
        n_finished_trajs = 0
        n_trans = 0
        while True:
            # ipdb.set_trace()
            action = policy(torch.tensor(observation, dtype=torch.float), deterministic=deterministic)[0].detach().numpy()
            # action = self._normalizer.unnormalize(action, "actions")
            # import ipdb
            # ipdb.set_trace()
            next_observation, reward, terminated, truncated, _ = self.envs.step(
                action + np.random.randn(action.shape[1]) * joint_noise_std + joint_noise_mean, ready_env_ids
            )
            done = np.logical_or(terminated, truncated)
            if self._render:
                getattr(self.envs, env_render_fn)()
                time.sleep(0.01)

            # next_observation = self._normalizer.normalize(
            #     next_observation, "observations"
            # )

            for idx, env_id in enumerate(ready_env_ids):
                observations[env_id].append(observation[idx])
                # import ipdb
                # ipdb.set_trace()
                actions[env_id].append(action[idx])
                rewards[env_id].append(reward[idx])
                next_observations[env_id].append(next_observation[idx])
                dones[env_id].append(done[idx])
            # import ipdb
            # ipdb.set_trace()
            if len(observations[0]) >= self.max_traj_length:

                ind = 0
                valid_length = np.argwhere(np.array(dones[0])==True)[0][0] + 1
                trajs.append(
                    dict(
                        observations=np.array(observations[ind][:valid_length], dtype=np.float32),
                        actions=np.array(actions[ind][:valid_length], dtype=np.float32),
                        rewards=np.array(rewards[ind][:valid_length], dtype=np.float32),
                        next_observations=np.array(
                            next_observations[ind][:valid_length], dtype=np.float32
                        ),
                        terminals=np.array(dones[ind][:valid_length], dtype=np.float32),
                    )
                )
                # import ipdb
                # ipdb.set_trace()
                n_trans += valid_length
                observations[ind] = []
                actions[ind] = []
                rewards[ind] = []
                next_observations[ind] = []
                dones[ind] = []

                n_finished_trajs += 1
                
                
                if n_finished_trajs % 100 == 0:
                    print(f'rollout num: {n_finished_trajs}')
                
                if n_trans > transition_num:
                    trajs = trajs
                    # ipdb.set_trace()
                    break
                # import ipdb
                # ipdb.set_trace()

                # surplus_env_num = len(ready_env_ids) - (n_trajs - n_finished_trajs)
                # if surplus_env_num > 0:
                #     mask = np.ones_like(ready_env_ids, dtype=bool)
                #     mask[env_ind_local[:surplus_env_num]] = False
                #     ready_env_ids = ready_env_ids[mask]
                # ipdb.set_trace()
                obs_reset, _ = self.envs.reset(np.arange(1))
                # obs_reset = self._normalizer.normalize(obs_reset, "observations")
                next_observation[np.arange(1)] = obs_reset

            elif np.any(done) or len(observations[0]) >= self.max_traj_length:
                pass
            #     env_ind_local = np.where(done)[0]
            #     env_ind_global = ready_env_ids[env_ind_local]

            #     for ind in env_ind_local:
            #         trajs.append(
            #             dict(
            #                 observations=np.array(observations[ind], dtype=np.float32),
            #                 actions=np.array(actions[ind], dtype=np.float32),
            #                 rewards=np.array(rewards[ind], dtype=np.float32),
            #                 next_observations=np.array(
            #                     next_observations[ind], dtype=np.float32
            #                 ),
            #                 terminals=np.array(dones[ind], dtype=np.float32),
            #             )
            #         )
            #         import ipdb
            #         ipdb.set_trace()
            #         n_trans += len(observations[ind])
            #         observations[ind] = []
            #         actions[ind] = []
            #         rewards[ind] = []
            #         next_observations[ind] = []
            #         dones[ind] = []
                    

            #     n_finished_trajs += len(env_ind_local)
            #     if n_trans > transition_num:
            #         trajs = trajs[:n_trajs]
            #         break

            #     # surplus_env_num = len(ready_env_ids) - (n_trajs - n_finished_trajs)
            #     # if surplus_env_num > 0:
            #     #     mask = np.ones_like(ready_env_ids, dtype=bool)
            #     #     mask[env_ind_local[:surplus_env_num]] = False
            #     #     ready_env_ids = ready_env_ids[mask]
            #     # ipdb.set_trace()
            #     obs_reset, _ = self.envs.reset(env_ind_global)
            #     # obs_reset = self._normalizer.normalize(obs_reset, "observations")
            #     next_observation[env_ind_local] = obs_reset
                
            
            
            observation = next_observation
        # import ipdb
        # ipdb.set_trace()
        return trajs


    @property
    def env(self):
        return self._env

    @property
    def envs(self):
        return self._envs

