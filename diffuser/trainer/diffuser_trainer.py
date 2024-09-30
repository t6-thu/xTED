import torch

from diffuser.algos import DecisionDiffuser
from diffuser.diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from diffuser.nets import DiffusionPlanner, InverseDynamic, RewardDynamic
from diffuser.policy import DiffuserPolicy
from diffuser.trainer.base_trainer import BaseTrainer
from utilities.data_utils import cycle, numpy_collate
from utilities.utils import set_random_seed, to_arch
from copy import deepcopy

import importlib
from utilities.model import TanhGaussianPolicy
import ipdb
import pickle
from viskit.logging import logger, setup_logger

import orbax

class DiffuserTrainer(BaseTrainer):
    def _setup(self, run_name : str):
        set_random_seed(self._cfgs.seed)
        # setup logger
        self._wandb_logger = self._setup_logger(run_name)

        # setup dataset and eval_sample
        train_dataset, eval_sampler, gen_sampler, tar_gen_sampler = self._setup_dataset()
        logger.save_SDEdit_data(train_dataset._data._dict, 'train_dataset')
        import ipdb
        # ipdb.set_trace()
        # _, gen_sampler = self._setup_dataset()
        # ipdb.set_trace()
        self._train_dataset = train_dataset
        data_sampler = torch.utils.data.RandomSampler(train_dataset)
        

        # setup policy
        self._planner, self._inv_model, self._reward_model = self._setup_policy()
        # ipdb.set_trace()
        # setup agent
        self._agent = DecisionDiffuser(
            self._cfgs.algo_cfg, self._planner, self._inv_model, self._reward_model
        )
        # ipdb.set_trace()
        # setup evaluator
        sampler_policy = DiffuserPolicy(self._planner, self._inv_model, self._reward_model)
        gen_sampler_policy = deepcopy(sampler_policy)
        # ipdb.set_trace()
        # gen_sampler_policy = deepcopy(sampler_policy)
        self._evaluator = self._setup_evaluator(sampler_policy, eval_sampler, train_dataset)
        self._sim_generator = self._setup_generator(None, gen_sampler, train_dataset, gen_sampler_policy)
        self._tar_generator = self._setup_generator(None, tar_gen_sampler, train_dataset, gen_sampler_policy)
        # ipdb.set_trace()
        self._online_dataset = None
        # online_data_sampler = torch.utils.data.RandomSampler(self._online_dataset)
        self._online_dataloader = None
        
        import ipdb

        # ipdb.set_trace()
        
        
        self._dataloader = cycle(
            torch.utils.data.DataLoader(
                train_dataset,
                sampler=data_sampler,
                batch_size=self._cfgs.batch_size,
                collate_fn=numpy_collate,
                drop_last=True,
                num_workers=8,
            )
        )
        # ipdb.set_trace()
        
        if self._cfgs.gen_trajs:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            # ckpt_path = '/home/chenqm/projects/ddit/logs/2024-03-22-14-33-59_halfcheetah-medium-expert-v2_N_attn12_hiddensize512_num_heads2_GOODDiT_MASK/diffuser_inv_d4rl/halfcheetah-medium-expert-v2/h_20-r_1200.0/100/checkpoints/model_1000'
            ckpt_path = self._cfgs['ckpt_path'].value
            # import ipdb
            # ipdb.set_trace()
            target = {'agent_states' : self._agent.train_states}
            restored = orbax_checkpointer.restore(ckpt_path, item=target)
            eval_params = {
                key: restored["agent_states"][key].params_ema or restored["agent_states"][key].params
                for key in self._agent.model_keys
            }
            # from D2C.sample import deploy_policy
            # from D2C.d2c.utils.utils import abs_file_path
            # file = abs_file_path(__file__, './D2C/example/benchmark/models/d4rl/mujoco/iql/HalfCheetah-v2_halfcheetah_medium_expert-v2/agent/0810/seed1/agent')
            # actor = deploy_policy(file)
            policy = TanhGaussianPolicy(
                # FIXME: fix hard code for state and action dims
                self._observation_dim,
                self._action_dim,
                arch='256-256',
                log_std_multiplier=1.,
                log_std_offset=-1.,
                orthogonal_init=False,
            )
            # model_path = '/home/jumpserver/nhy/ddit/model/actor.pth' # haomo path
            # model_path = '/dysData/DISCOVER/nhy/ddit/model/maze2d_actor.pth' # discover path
            # model_path = '/home/chenqm/projects/ddit/h_2_o/actor.pth'

            if 'walker' in self._cfgs.env:
                if self._cfgs.dynamic == 'joint_noise':
                    model_path = f'./h_2_o/walker_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_{self._cfgs.std}.pth' 
                else:
                    model_path = f'./h_2_o/walker_{self._cfgs.dynamic}x{self._cfgs.variety_degree}.pth' 
            elif 'halfcheetah' in self._cfgs.env:
                if self._cfgs.dynamic == 'joint_noise':
                    model_path = f'./h_2_o/halfcheetah_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_{self._cfgs.std}.pth' 
                else:
                    model_path = f'./h_2_o/halfcheetah_{self._cfgs.dynamic}x{self._cfgs.variety_degree}.pth' 
            elif 'hopper' in self._cfgs.env:
                if self._cfgs.dynamic == 'joint_noise':
                    model_path = f'./h_2_o/hopper_{self._cfgs.dynamic}x{self._cfgs.variety_degree}_{self._cfgs.std}.pth' 
                else:
                    model_path = f'./h_2_o/hopper_{self._cfgs.dynamic}x{self._cfgs.variety_degree}.pth' 
            elif 'maze2d' in self._cfgs.env:
                # model_path = f'./h_2_o/maze2d_{self._cfgs.dynamic}x{self._cfgs.variety_degree}.pth' 
                model_path = f'./model/maze2d_actor.pth'

            # import ipdb
            # ipdb.set_trace()

            # model_path = '/home/chenqm/projects/ddit/h_2_o/xml_path/actor.pth'
            policy.load_state_dict(torch.load(model_path))
            # ipdb.set_trace()
            self._sim_generator.update_policy(policy)
            self._tar_generator.update_policy(policy)
            self._sim_generator._diffuser.update_params(eval_params)
            self._tar_generator._diffuser.update_params(eval_params)
        
        
        
        
    def _setup_policy(self):
        gd = GaussianDiffusion(
            num_timesteps=self._cfgs.algo_cfg.num_timesteps,
            schedule_name=self._cfgs.algo_cfg.schedule_name,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            returns_condition=self._cfgs.returns_condition,
            condition_guidance_w=self._cfgs.condition_guidance_w,
            sample_temperature=self._cfgs.algo_cfg.sample_temperature,
            OU_mu = self._cfgs['mu'].value,
            OU_sigma = self._cfgs['sigma'].value,
            OU_theta = self._cfgs['theta'].value,
            noise_type = self._cfgs['noise_type'].value,
            use_condition = self._cfgs['use_condition'].value
        )

        
        plan_sample_dim = self._observation_dim
        if self._cfgs.use_inv_dynamic:
            inv_model = InverseDynamic(
                action_dim=self._action_dim,
                hidden_dims=to_arch(self._cfgs.inv_hidden_dims),
            )
            
            plan_action_dim = 0
        else:
            inv_model = None
            plan_action_dim = self._action_dim
            plan_sample_dim += self._action_dim

        if self._cfgs.use_reward_dynamic:
            reward_model = RewardDynamic(
                reward_dim=self._reward_dim,
                hidden_dims=to_arch(self._cfgs.reward_hidden_dims)
            )
            plan_reward_dim = 0
        else:
            reward_model = None
            plan_reward_dim = self._reward_dim
            plan_sample_dim += self._reward_dim

        # elif self._cfgs.algo_cfg.edit_sar:
        #     inv_model = None
        #     reward_model = None
        #     plan_sample_dim = self._observation_dim + self._action_dim + 1
        #     plan_action_dim = self._action_dim
        # else:
        #     inv_model = None
        #     reward_model = None
        #     plan_sample_dim = self._observation_dim + self._action_dim
        #     plan_action_dim = self._action_dim

        planner = DiffusionPlanner(
            cfgs=self._cfgs,
            diffusion=gd,
            horizon=self._cfgs.horizon,
            sample_dim=plan_sample_dim,
            observation_dim=self._observation_dim,
            action_dim=plan_action_dim,
            reward_dim=plan_reward_dim,
            dim=self._cfgs.dim,
            dim_mults=to_arch(self._cfgs.dim_mults),
            returns_condition=self._cfgs.returns_condition,
            condition_dropout=self._cfgs.condition_dropout,
            kernel_size=self._cfgs.kernel_size,
            sample_method=self._cfgs.sample_method,
            dpm_steps=self._cfgs.algo_cfg.dpm_steps,
            dpm_t_end=self._cfgs.algo_cfg.dpm_t_end,
            inv_dynamics = True if inv_model is not None else False,
            reward_dynamics = True if reward_model is not None else False,
            use_condition=self._cfgs.use_condition,
            use_goal_condition=self._cfgs.use_goal_condition,
            seperate_encoding = self._cfgs.seperate_encoding,
        )
        return planner, inv_model, reward_model
