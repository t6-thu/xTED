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

import importlib

import absl
import absl.flags
import gym
import jax
import jax.numpy as jnp
import torch
import tqdm

from diffuser.constants import DATASET, DATASET_MAP, ENV_MAP
from diffuser.hps import hyperparameters
from utilities.data_utils import cycle, numpy_collate
from utilities.jax_utils import batch_to_jax
from utilities.normalization import DatasetNormalizer
from utilities.sampler import TrajSampler, StepSampler, OnlineTrajSampler, MazeSampler
from env.maze_continual import Maze
from utilities.utils import (
    DotFormatter,
    Timer,
    WandBLogger,
    get_user_flags,
    prefix_metrics,
)
from viskit.logging import logger, setup_logger

import ipdb
from copy import deepcopy

from data.d4rl import OnlineDataset
import numpy as np

from torch.utils.data import ConcatDataset
from itertools import chain

import pickle
import wandb

class BaseTrainer:
    def __init__(self, config, use_absl: bool = True, run_name = ''):
        if use_absl:
            self._cfgs = absl.flags.FLAGS
        else:
            self._cfgs = config

        self._cfgs.algo_cfg.max_grad_norm = hyperparameters[self._cfgs.env]["gn"]
        self._cfgs.algo_cfg.lr_decay_steps = (
            self._cfgs.n_epochs * self._cfgs.n_train_step_per_epoch
        )

        if self._cfgs.activation == "mish":
            act_fn = lambda x: x * jnp.tanh(jax.nn.softplus(x))
        else:
            act_fn = getattr(jax.nn, self._cfgs.activation)

        self._act_fn = act_fn
        self._variant = get_user_flags(self._cfgs, config)
        self.run_name = run_name
        # get high level env
        env_name_full = self._cfgs.env
        for scenario_name in ENV_MAP:
            if scenario_name in env_name_full:
                self._env = ENV_MAP[scenario_name]
                break
        else:
            raise NotImplementedError

    def train(self, opt = 'train', t_add = 0.05, t_de = 0.05, source = 'target'):
        if opt == 'train':
            # ipdb.set_trace()
            self._setup(self.run_name)
            # ipdb.set_trace()
            # for debug
            # self._online_dataset = self._gen_dataset(self._generator)
            # online_data_sampler = torch.utils.data.RandomSampler(self._online_dataset)
            # self._online_dataloader = cycle(
            #     torch.utils.data.DataLoader(
            #         self._online_dataset,
            #         sampler=online_data_sampler,
            #         batch_size=self._cfgs.batch_size,
            #         collate_fn=numpy_collate,
            #         drop_last=True,
            #         num_workers=8,
            #     )
            # )
            # self._dataloader = self._online_dataloader

                        
            viskit_metrics = {}
            for epoch in range(self._cfgs.n_epochs):
                metrics = {"epoch": epoch}
                
                # ipdb.set_trace()
                # self.test()
                # self._gen_dataset()
                # ipdb.set_trace()
                
                with Timer() as eval_timer:
                    if self._cfgs.eval_period > 0 and epoch % self._cfgs.eval_period == 0:
                        self._evaluator.update_params(self._agent.eval_params)
                        # eval_metrics = self._evaluator.evaluate(epoch)
                        
                        # metrics.update(eval_metrics)

                    if self._cfgs.save_period > 0 and epoch % self._cfgs.save_period == 0:
                        self._save_model(epoch)

                with Timer() as train_timer:
                    for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
                        import ipdb
                        # ipdb.set_trace()
                        next(self._dataloader)
                        batch = batch_to_jax(next(self._dataloader))
                        # import ipdb
                        # ipdb.set_trace()

                        # ipdb.set_trace()
                        metrics.update(prefix_metrics(self._agent.train(batch), "agent"))
                        # ipdb.set_trace()
                    # ipdb.set_trace()
                    
                
                with Timer() as gen_timer:
                    if self._cfgs.gen_trajs:   
                        if self._cfgs.gen_period > 0 and epoch % self._cfgs.gen_period == 0:
                        # if True:
                            # self._generator.update_params(self._agent.eval_params)
                            if self._online_dataset is None:
                                self._online_dataset = self._gen_dataset()
                                # ipdb.set_trace()
                            else:
                                
                                if len(self._online_dataset) < self._cfgs.gen_max_size:
                                    self._online_dataset = ConcatDataset([self._online_dataset, self._gen_dataset()])
                            online_data_sampler = torch.utils.data.RandomSampler(self._online_dataset)
                            self._online_dataloader = cycle(
                                torch.utils.data.DataLoader(
                                    self._online_dataset,
                                    sampler=online_data_sampler,
                                    batch_size=self._cfgs.batch_size,
                                    collate_fn=numpy_collate,
                                    drop_last=True,
                                    num_workers=8,
                                )
                            )
                            self._dataloader = chain(self._dataloader, self._online_dataloader)

                metrics["train_time"] = train_timer()
                metrics["eval_time"] = eval_timer()
                metrics["gen_time"] = gen_timer()
                metrics["epoch_time"] = train_timer() + eval_timer() + gen_timer()
                self._wandb_logger.log(metrics)
                viskit_metrics.update(metrics)
                logger.record_dict(viskit_metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)

            # save model at final epoch
            if self._cfgs.save_period > 0 and self._cfgs.n_epochs % self._cfgs.save_period == 0:
                self._save_model(self._cfgs.n_epochs)

            if self._cfgs.eval_period > 0 and self._cfgs.n_epochs % self._cfgs.eval_period == 0:
                self._evaluator.update_params(self._agent.eval_params)
                # self._evaluator.evaluate(self._cfgs.n_epochs)
        else:
            self._setup(self.run_name)
            self.gen(t_add = int(t_add * self._cfgs.algo_cfg.num_timesteps), t_de = int(t_de * self._cfgs.algo_cfg.num_timesteps), source=source)

    def _setup(self, run_name):
        raise NotImplementedError

    def _save_model(self, epoch: int):
        save_data = {
            "agent_states": self._agent.train_states,
            "variant": self._variant,
            "epoch": epoch,
        }
        logger.save_orbax_checkpoint(
            save_data, f"checkpoints/model_{epoch}"
        )

    def _setup_logger(self, run_name):
        logging_configs = self._cfgs.logging
        logging_configs["log_dir"] = DotFormatter().vformat(
            self._cfgs.log_dir_format, [], self._variant
        )
        wandb_logger = WandBLogger(config=logging_configs, variant=self._variant, run_name = run_name)
        setup_logger(
            variant=self._variant,
            log_dir=wandb_logger.output_dir,
            seed=self._cfgs.seed,
            include_exp_prefix_sub_dir=False,
        )
        return wandb_logger

    def _setup_d4rl(self):
        from data.d4rl import get_dataset

        if self._cfgs.dataset_class in ["QLearningDataset"]:
            include_next_obs = True
        else:
            include_next_obs = False
            
        from utilities.utils import get_new_gravity_env, get_new_friction_env, get_new_wind_env, get_new_thigh_env, get_new_torso_env
        # eval_sampler = MazeSampler(
        #     Maze(),
        #     self._cfgs.max_gen_traj_length,
        # )
        
        # gen_sampler = MazeSampler(
        #     Maze(),
        #     self._cfgs.max_gen_traj_length,
        # )
        
        # tar_gen_sampler = MazeSampler(
        #     Maze(),
        #     self._cfgs.max_gen_traj_length,
        # )
        if self._cfgs.dynamic == 'gravity':
            eval_sampler = TrajSampler(
                lambda: get_new_gravity_env(1, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_traj_length,
            )
            
            gen_sampler = OnlineTrajSampler(
                lambda: get_new_gravity_env(self._cfgs.variety_degree, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
            
            tar_gen_sampler = OnlineTrajSampler(
                lambda: get_new_gravity_env(1.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
        elif self._cfgs.dynamic == 'friction':
            eval_sampler = TrajSampler(
                lambda: get_new_friction_env(1, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_traj_length,
            )
            
            gen_sampler = OnlineTrajSampler(
                lambda: get_new_friction_env(self._cfgs.variety_degree, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
            
            tar_gen_sampler = OnlineTrajSampler(
                lambda: get_new_friction_env(1.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
        elif self._cfgs.dynamic == 'thigh_size':
            eval_sampler = TrajSampler(
                lambda: get_new_thigh_env(1, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_traj_length,
            )
            
            gen_sampler = OnlineTrajSampler(
                lambda: get_new_thigh_env(self._cfgs.variety_degree, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
            
            tar_gen_sampler = OnlineTrajSampler(
                lambda: get_new_thigh_env(1.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
        elif self._cfgs.dynamic == 'torso_length':
            eval_sampler = TrajSampler(
                lambda: get_new_torso_env(1, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_traj_length,
            )
            
            gen_sampler = OnlineTrajSampler(
                lambda: get_new_torso_env(self._cfgs.variety_degree, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
            
            tar_gen_sampler = OnlineTrajSampler(
                lambda: get_new_torso_env(1.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
             
        elif self._cfgs.dynamic == 'wind_x':
            eval_sampler = TrajSampler(
                lambda: get_new_wind_env(0.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_traj_length,
            )
            
            gen_sampler = OnlineTrajSampler(
                lambda: get_new_wind_env(self._cfgs.variety_degree, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
            
            tar_gen_sampler = OnlineTrajSampler(
                lambda: get_new_wind_env(0.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
        elif self._cfgs.dynamic == 'joint_noise':
            eval_sampler = TrajSampler(
                lambda: get_new_gravity_env(1.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_traj_length,
            )
            
            gen_sampler = OnlineTrajSampler(
                lambda: get_new_gravity_env(1.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
            
            tar_gen_sampler = OnlineTrajSampler(
                lambda: get_new_gravity_env(1.0, self._cfgs.env),
                self._cfgs.num_gen_envs,
                self._cfgs.eval_env_seed,
                self._cfgs.max_gen_traj_length,
            )
        
        
        dataset = get_dataset(
            self._cfgs,
            eval_sampler.env,
            f'./gen_real_data/small_samples/{self._cfgs.env}-ratio-{self._cfgs.ratio}.npy',
            max_traj_length=self._cfgs.max_traj_length,
            include_next_obs=include_next_obs,
            termination_penalty=self._cfgs.termination_penalty,
        )
        
        
        t = np.mean(dataset['observations'][:,:,0])
        return dataset, eval_sampler, gen_sampler, tar_gen_sampler
    

    def _setup_dataset(self):
        dataset_type = DATASET_MAP[self._cfgs.dataset]
        if dataset_type == DATASET.D4RL:
            dataset, eval_sampler, gen_sampler, tar_gen_sampler = self._setup_d4rl()
        else:
            raise NotImplementedError
        # ipdb.set_trace()
        train_dataset = deepcopy(dataset)

                    
        train_dataset["rewards"] = (
            train_dataset["rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
        ) # this is not right
        
        # ipdb.set_trace()
        train_dataset = getattr(
            importlib.import_module("data.sequence"), self._cfgs.dataset_class
        )(
            train_dataset,
            horizon=self._cfgs.horizon,
            max_traj_length=self._cfgs.max_traj_length,
            include_returns=self._cfgs.include_returns,
            normalizer=self._cfgs.normalizer,
            returns_scale=self._cfgs.returns_scale,
            use_padding=self._cfgs.use_padding,
            edit_sar=self._cfgs.algo_cfg.edit_sar,
        )

        eval_sampler.set_normalizer(train_dataset.normalizer)
        gen_sampler.set_normalizer(train_dataset.normalizer)
        tar_gen_sampler.set_normalizer(train_dataset.normalizer)

        # import ipdb
        # ipdb.set_trace()
        self._observation_dim = eval_sampler.env.observation_space.shape[0]
        self._action_dim = eval_sampler.env.action_space.shape[0]
        self._reward_dim = 1

        return train_dataset, eval_sampler, gen_sampler, tar_gen_sampler

    def _setup_evaluator(self, sampler_policy, eval_sampler, dataset):
        evaluator_class = getattr(
            importlib.import_module("diffuser.evaluator"), self._cfgs.evaluator_class
        )

        if evaluator_class.eval_mode == "online":
            evaluator = evaluator_class(self._cfgs, sampler_policy, eval_sampler)
        elif evaluator_class.eval_mode == "offline":
            eval_data_sampler = torch.utils.data.RandomSampler(dataset)
            eval_dataloader = cycle(
                torch.utils.data.DataLoader(
                    dataset,
                    sampler=eval_data_sampler,
                    batch_size=self._cfgs.eval_batch_size,
                    collate_fn=numpy_collate,
                    drop_last=True,
                    num_workers=4,
                )
            )
            evaluator = evaluator_class(self._cfgs, sampler_policy, eval_dataloader)
        elif evaluator_class.eval_mode == "skip":
            evaluator = evaluator_class(self._cfgs, sampler_policy)
        else:
            raise NotImplementedError(f"Unknown eval_mode: {self._cfgs.eval_mode}")

        return evaluator
    
    def _setup_generator(self, sampler_policy, gen_sampler, dataset, diffuser):
        generator_class = getattr(
            importlib.import_module("diffuser.evaluator"), self._cfgs.generator_class
        )

        if generator_class.eval_mode == "online":
            generator = generator_class(self._cfgs, sampler_policy, gen_sampler, diffuser)
        elif evaluator_class.eval_mode == "offline":
            eval_data_sampler = torch.utils.data.RandomSampler(dataset)
            eval_dataloader = cycle(
                torch.utils.data.DataLoader(
                    dataset,
                    sampler=eval_data_sampler,
                    batch_size=self._cfgs.eval_batch_size,
                    collate_fn=numpy_collate,
                    drop_last=True,
                    num_workers=4,
                )
            )
            evaluator = evaluator_class(self._cfgs, sampler_policy, eval_dataloader)
        elif evaluator_class.eval_mode == "skip":
            evaluator = evaluator_class(self._cfgs, sampler_policy)
        else:
            raise NotImplementedError(f"Unknown eval_mode: {self._cfgs.eval_mode}")
        return generator
        
    

    def merge_dicts(self, trajs):
        import numpy as np
        merged_dict = {}
        keys = trajs[0].keys()
        # ipdb.set_trace()
        for key in keys:
            # 获取当前键对应的所有元素列表
            elements = [d[key] for d in trajs]
            # 获取每个元素列表的最大长度
            # ipdb.set_trace()
            max_length = max(len(arr) for arr in elements)
            # ipdb.set_trace()
            x = elements[0].shape[-1] if len(elements[0].shape) > 1 else 0  # 第二维的大小
            # ipdb.set_trace()
            # 创建一个空的数组来存储合并后的结果
            merged_array = np.empty((len(trajs), max_length, x)) if x > 0 else np.empty((len(trajs), max_length))
            merged_array.fill(np.nan)
            # ipdb.set_trace()
            for i, arr in enumerate(elements):
                if x > 0:
                    merged_array[i, :len(arr), :] = arr
                else:
                    merged_array[i, :len(arr)] = arr
            # ipdb.set_trace()
            # 将合并后的数组赋值给字典的对应键
            merged_dict[key] = merged_array

        return merged_dict
    
    def gen(self, t_add = 0.05, t_de = 0.05, source = 'target'):
        import ipdb
        if not self._cfgs.gen_trajs:
            return None
        
        
        normalizer = self._sim_generator._gen_sampler._normalizer
        # ipdb.set_trace()
        if source == 'target':
            import ipdb
            # ipdb.set_trace()
            length = len(self._train_dataset._data['observations'])
            real_obs = []
            real_act = []
            real_rewards = []
            for j in range(length):
                # import ipdb
                # ipdb.set_trace()
                valid_idx = int(self._train_dataset._data['traj_lengths'][j] // self._cfgs.horizon) * self._cfgs.horizon
                obs = self._train_dataset._data['observations'][j][:valid_idx,:].reshape((-1, self._cfgs.horizon, self._observation_dim), order='C')
                act = self._train_dataset._data['actions'][j][:valid_idx,:].reshape((-1, self._cfgs.horizon, self._action_dim), order='C')
                reward = self._train_dataset._data['rewards'][j][:valid_idx].reshape((-1, self._cfgs.horizon, self._reward_dim), order='C')
                real_obs.append(obs)
                real_act.append(act)
                real_rewards.append(reward)
            # import ipdb
            # ipdb.set_trace()
            real_obs = np.concatenate([obs for obs in real_obs])
            real_act = np.concatenate([act for act in real_act])
            real_rewards = np.concatenate([reward for reward in real_rewards])
            # real_obs = self._train_dataset._data['observations'].reshape((-1, self._cfgs.horizon, self._observation_dim), order='C')

            # real_act = self._train_dataset._data['actions'].reshape((-1, self._cfgs.horizon, self._action_dim), order='C')

            # real_rewards = self._train_dataset._data['rewards'].reshape((-1, self._cfgs.horizon, self._reward_dim), order='C')
            trajs = list()
            # ipdb.set_trace()
            for i in range(real_act.shape[0]):
                try:
                    trajs.append(dict(
                        observations=None,
                        actions=None,
                        next_observations=None,
                        rewards=None,
                        terminals=None
                    ))
                    trajs[i]['observations'] = real_obs[i]
                    trajs[i]['actions'] = real_act[i]
                    trajs[i]['rewards'] = real_rewards[i]

                except:
                    ipdb.set_trace()
            ipdb.set_trace
            logger.save_SDEdit_data(trajs, 'unEdited')
            
        else:
            
            
            if 'walker' in self._cfgs.env:
                if 'replay' in self._cfgs.env:
                    if self._cfgs.ratio != 15.0:
                        model_path = f'./h_2_o/Walker2d-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_ratio{self._cfgs.ratio}_unEdited.npy'
                    else:
                        if self._cfgs.dynamic == 'joint_noise':
                            model_path = f'./h_2_o/Walker2d-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_{self._cfgs.std}_sr{self._cfgs.sim_ratio}_unEdited.npy' 
                        else:
                            model_path = f'./h_2_o/Walker2d-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_sr{self._cfgs.sim_ratio}_unEdited.npy' 

                else:
                    if self._cfgs.dynamic == 'joint_noise':
                        model_path = f'./h_2_o/Walker2d-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_{self._cfgs.std}_sr{self._cfgs.sim_ratio}_unEdited.npy' 

                    else:
                        model_path = f'./h_2_o/Walker2d-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_sr{self._cfgs.sim_ratio}_unEdited.npy' 

            elif 'halfcheetah' in self._cfgs.env:
                if self._cfgs.dynamic == 'joint_noise':
                     model_path = f'./h_2_o/Halfcheetah-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_{self._cfgs.std}_sr{self._cfgs.sim_ratio}_unEdited.npy' 
                else:
                    model_path = f'./h_2_o/Halfcheetah-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_sr{self._cfgs.sim_ratio}_unEdited.npy' 
            elif 'hopper' in self._cfgs.env:
                if self._cfgs.dynamic == 'joint_noise':
                    model_path = f'./h_2_o/Hopper-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_{self._cfgs.std}_sr{self._cfgs.sim_ratio}_unEdited.npy' 
                else:
                    model_path = f'./h_2_o/Hopper-v2_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_sr{self._cfgs.sim_ratio}_unEdited.npy' 
            elif 'maze2d' in self._cfgs.env:
                # model_path = f'./model/maze2d_{self._cfgs.dynamic}x{self._cfgs.variety_degree}.pth' 
                model_path = f'./h_2_o/maze2d_actor.pth'
            import os
            # ipdb.set_trace()
            if not os.path.exists(model_path):
                # import ipdb
                # ipdb.set_trace()
                num_gen = int((self._cfgs.sim_ratio * self._train_dataset._data.env_data_length * self._cfgs.split_rate) // self._cfgs.max_gen_traj_length)

                
                trajs = self._sim_generator.generate(num_gen, joint_noise_mean = self._cfgs.variety_degree if self._cfgs.dynamic == 'joint_noise' else 0., joint_noise_std = self._cfgs.std)
                # ipdb.set_trace()
                # trajs = []
                # trajs.append(dict(
                #     observations = np.zeros((998,17)),
                #     actions = np.zeros((998,6)),
                #     rewards = np.zeros((998)),
                #     terminals = np.zeros((998)),
                # ))
                # trajs[0]['terminals'][-1] = 1
                # import ipdb
                # ipdb.set_trace()
                noisy_trajs = deepcopy(trajs)
                
                
                new_trajs = []
                bias = 0
                for i in range(len(trajs)):
                    for k in range(self._cfgs.gen_num):
                        new_trajs.append(dict())
                    
                    for key in trajs[i].keys():
                        # import ipdb
                        # ipdb.set_trace()
                        try:
                            traj_length = len(trajs[i][key])
                            valid_length = self._cfgs.horizon * int(traj_length / self._cfgs.horizon)
                            if valid_length == 0 :
                                continue
                            if key == 'terminals':
                                trajs[i][key][valid_length-1] = 1
                            trajs[i][key] = trajs[i][key][:valid_length,]
                            trajs[i][key] = trajs[i][key].reshape(-1, self._cfgs.horizon, trajs[i][key].shape[-1]) if  len(trajs[i][key].shape) > 1 else trajs[i][key].reshape(-1, self._cfgs.horizon, 1)
                            temp = np.split(trajs[i][key], trajs[i][key].shape[0])
                            # if key == 'rewards':
                            #     import ipdb
                            #     ipdb.set_trace()
                            for j in range(len(temp)):
                                new_trajs[bias + j][key] = np.squeeze(temp[j], axis=0)
                        except:
                            import ipdb
                            ipdb.set_trace()
                    bias += self._cfgs.gen_num
                trajs = new_trajs
                data = trajs
                new_data = dict()
                for key in data[0].keys():
                    new_data[key] = []

                for d in data:
                    # d is a dict
                    for key in d.keys():
                        new_data[key].append(d[key])
                for key in new_data.keys():
                    new_data[key] = np.array(new_data[key]).reshape((-1,new_data[key][0].shape[-1])) if (new_data[key][0].shape[-1] > 1) else np.array(new_data[key]).reshape((-1,new_data[key][0].shape[-1])).squeeze()
                # import ipdb
                # ipdb.set_trace()  
                np.save(model_path, new_data)
            # unEdited_data = logger.copy_load_SDEdit_data(model_path, 'unEdited', load=True)
            
            from data.d4rl import get_dataset

            if self._cfgs.dataset_class in ["QLearningDataset"]:
                include_next_obs = True
            else:
                include_next_obs = False
                
            from utilities.utils import get_new_gravity_env, get_new_friction_env, get_new_wind_env, get_new_torso_env, get_new_thigh_env
            
            if self._cfgs.dynamic == 'gravity':
                eval_sampler = OnlineTrajSampler(
                    lambda: get_new_gravity_env(1.0, self._cfgs.env),
                    self._cfgs.num_gen_envs,
                    self._cfgs.eval_env_seed,
                    self._cfgs.max_gen_traj_length,
                )
            elif self._cfgs.dynamic == 'friction':          
                eval_sampler = OnlineTrajSampler(
                    lambda: get_new_friction_env(1.0, self._cfgs.env),
                    self._cfgs.num_gen_envs,
                    self._cfgs.eval_env_seed,
                    self._cfgs.max_gen_traj_length,

                )
            elif self._cfgs.dynamic == 'thigh_size':          
                eval_sampler = OnlineTrajSampler(
                    lambda: get_new_thigh_env(1.0, self._cfgs.env),
                    self._cfgs.num_gen_envs,
                    self._cfgs.eval_env_seed,
                    self._cfgs.max_gen_traj_length,

                )
            elif self._cfgs.dynamic == 'torso_length':          
                eval_sampler = OnlineTrajSampler(
                    lambda: get_new_torso_env(1.0, self._cfgs.env),
                    self._cfgs.num_gen_envs,
                    self._cfgs.eval_env_seed,
                    self._cfgs.max_gen_traj_length,

                )
            elif self._cfgs.dynamic == 'wind_x':
                eval_sampler = TrajSampler(
                    lambda: get_new_wind_env(0., self._cfgs.env),
                    self._cfgs.num_gen_envs,
                    self._cfgs.eval_env_seed,
                    self._cfgs.max_traj_length,
                )
            elif self._cfgs.dynamic == 'joint_noise':
                eval_sampler = TrajSampler(
                    lambda: get_new_torso_env(1., self._cfgs.env),
                    self._cfgs.num_gen_envs,
                    self._cfgs.eval_env_seed,
                    self._cfgs.max_traj_length,
                )
            
            unEdited_dataset = get_dataset(
                self._cfgs,
                eval_sampler.env,
                model_path,
                max_traj_length=self._cfgs.max_traj_length,
                include_next_obs=include_next_obs,
                termination_penalty=self._cfgs.termination_penalty,
            )
            
            # import ipdb
            # ipdb.set_trace()
            # normalizer = DatasetNormalizer(
            #     unEdited_dataset,
            #     self._cfgs.normalizer
            # )
            length = len(unEdited_dataset['observations'])
            real_obs = []
            real_act = []
            real_rewards = []
            for j in range(length):
                # import ipdb
                # ipdb.set_trace()
                valid_idx = int(unEdited_dataset['traj_lengths'][j] // self._cfgs.horizon) * self._cfgs.horizon
                obs = unEdited_dataset['observations'][j][:valid_idx,:].reshape((-1, self._cfgs.horizon, self._observation_dim), order='C')
                act = unEdited_dataset['actions'][j][:valid_idx,:].reshape((-1, self._cfgs.horizon, self._action_dim), order='C')
                reward = unEdited_dataset['rewards'][j][:valid_idx].reshape((-1, self._cfgs.horizon, self._reward_dim), order='C')
                real_obs.append(obs)
                real_act.append(act)
                real_rewards.append(reward)
            # import ipdb
            # ipdb.set_trace()
            real_obs = np.concatenate([obs for obs in real_obs])
            real_act = np.concatenate([act for act in real_act])
            real_rewards = np.concatenate([reward for reward in real_rewards])
            # real_obs = unEdited_dataset['observations'].reshape((-1, self._cfgs.horizon, self._observation_dim), order='C')

            # real_act = unEdited_dataset['actions'].reshape((-1, self._cfgs.horizon, self._action_dim), order='C')

            # real_rewards = unEdited_dataset['rewards'].reshape((-1, self._cfgs.horizon, self._reward_dim), order='C')
            trajs = list()
            # ipdb.set_trace()
            for i in range(real_act.shape[0]):
                try:
                    trajs.append(dict(
                        observations=None,
                        actions=None,
                        next_observations=None,
                        rewards=None,
                        terminals=None
                    ))
                    trajs[i]['observations'] = real_obs[i]
                    trajs[i]['actions'] = real_act[i]
                    trajs[i]['rewards'] = real_rewards[i]

                except:
                    ipdb.set_trace()
            logger.save_SDEdit_data(trajs, 'unEdited')
            
            # import ipdb
            # ipdb.set_trace()
        for i in tqdm.trange(len(trajs)):
            # ipdb.set_trace()
            # import ipdb
            # ipdb.set_trace()
            for e in range(self._cfgs.edit_time):
                # import ipdb
                # ipdb.set_trace()
                if e == 0:
                    trajs[i]['observations'], trajs[i]['actions'], trajs[i]['rewards'], _, _ = self._sim_generator._diffuser(
                        observations = normalizer.normalize(trajs[i]['observations'], 'observations'),
                        actions = normalizer.normalize(trajs[i]['actions'], 'actions'),
                        rewards = normalizer.normalize(trajs[i]['rewards'], 'rewards'),
                        choice = 'gen',
                        t_add = t_add,
                        t_de = t_de,

                    )
                else:
                    trajs[i]['observations'], trajs[i]['actions'], trajs[i]['rewards'], _, _ = self._sim_generator._diffuser(
                        observations = normalizer.normalize(trajs[i]['observations'][0], 'observations'),
                        actions = normalizer.normalize(trajs[i]['actions'][0], 'actions'),
                        rewards = normalizer.normalize(trajs[i]['rewards'][0], 'rewards'),
                        choice = 'gen',
                        t_add = t_add,
                        t_de = t_de,

                    )
                # ipdb.set_trace()
                trajs[i]['observations'] = normalizer.unnormalize(trajs[i]['observations'], 'observations')
                trajs[i]['actions'] = normalizer.unnormalize(trajs[i]['actions'], 'actions')
                trajs[i]['rewards'] = normalizer.unnormalize(trajs[i]['rewards'], 'rewards')
                trajs[i]['next_observations'] = np.concatenate(
                    [
                        trajs[i]['observations'][:,1:,],
                        trajs[i]['observations'][:,-1:,]
                    ], axis=1
                )
            
        
        logger.save_SDEdit_data(trajs, 'Edited')

        return
        
        
    
    def _gen_dataset(self):
        if not self._cfgs.gen_trajs:
            return None
        
        
        # generator.update_params(self._agent.eval_params)
        ipdb.set_trace()
        # normalizer = self._sim_generator._gen_sampler._normalizer
        # real_obs = self._dataset._data['observations'].reshape((1002*50, 20, 18))
        # real_act = self._dataset._data['actions'].reshape((1002*50, 20, 6))
        trajs = self._sim_generator.generate(self._cfgs.gen_n_trajs)
        noisy_trajs = deepcopy(trajs)
        tar_trajs = self._tar_generator.generate(self._cfgs.gen_n_trajs)
        # trajs = generator.generate(1)
        # ipdb.set_trace()
        # merged_traj = self.merge_dicts(trajs)
        merged_traj = {}
        
        for traj in trajs:
            for key, value in traj.items():
                if key in merged_traj:
                    merged_traj[key].append(value)
                else:
                    merged_traj[key] = [value] 
                              
        for key, value_list in merged_traj.items():
            merged_traj[key] = np.concatenate(value_list, axis=0)
            merged_traj[key] = merged_traj[key].reshape((1, *merged_traj[key].shape))
        # # ipdb.set_trace()
        
        # conditions = {0: observations}
        # returns = jnp.ones((observations.shape[0], 1)) * 0.9
        
        # TODO: calculate reward
        logger.save_SDEdit_data(trajs, 'unEdited')
        logger.save_SDEdit_data(tar_trajs, 'tar')
        online_dataset = None
        ipdb.set_trace()
        for i in tqdm.trange(len(trajs)):
            x_pos = trajs[i]['observations'][:,-1]
            delta_x_pos = x_pos[1:] - x_pos[:-1]
            delta_x_pos = np.concatenate(
                [
                    delta_x_pos,
                    delta_x_pos[-1:]
                ]
            )
            
            trajs[i]['observations'][:,-1] = delta_x_pos
            trajs[i]['observations'], trajs[i]['actions'], noisy_trajs[i]['observations'], noisy_trajs[i]['actions'] = self._sim_generator._diffuser(
                observations = normalizer.normalize(trajs[i]['observations'], 'observations'),
                actions = normalizer.normalize(trajs[i]['actions'], 'actions'),
                choice = 'gen'
            )
            # ipdb.set_trace()
            trajs[i]['observations'] = normalizer.unnormalize(trajs[i]['observations'], 'observations')
            trajs[i]['actions'] = normalizer.unnormalize(trajs[i]['actions'], 'actions')
            trajs[i]['next_observations'] = np.concatenate(
                [
                    trajs[i]['observations'][:,1:,],
                    trajs[i]['observations'][:,-1:,]
                ], axis=1
            )
            delta_x = trajs[i]['observations'][:,:,-1]
            rewards_ctrl = -0.1 * np.square(trajs[i]['actions']).sum(axis=2)
            rewards_run = delta_x / 0.05
            rewards = (rewards_run + rewards_ctrl).reshape(-1)
            # ipdb.set_trace()
            trajs[i]['rewards'] = rewards
            
            online_dataset = OnlineDataset(trajs[i]) if online_dataset is None else ConcatDataset([online_dataset, OnlineDataset(trajs[i])])
        
        logger.save_SDEdit_data(trajs, 'Edited')
        logger.save_SDEdit_data(noisy_trajs, 'noisy')
        # online_dataset = OnlineDataset(merged_traj)
        # ipdb.set_trace()
        return online_dataset
        
