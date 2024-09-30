import numpy as np
import torch

from diffuser.algos import DiffusionQL
from diffuser.diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from diffuser.nets import Critic, DiffusionPolicy, GaussianPolicy, Value
from diffuser.policy import SamplerPolicy
from diffuser.trainer.base_trainer import BaseTrainer
from utilities.data_utils import cycle, numpy_collate
from utilities.utils import set_random_seed, to_arch


class DiffusionQLTrainer(BaseTrainer):
    def _setup(self):
        set_random_seed(self._cfgs.seed)
        # setup logger
        self._wandb_logger = self._setup_logger()

        # setup dataset and eval_sample
        dataset, eval_sampler = self._setup_dataset()
        data_sampler = torch.utils.data.RandomSampler(dataset)
        self._dataloader = cycle(
            torch.utils.data.DataLoader(
                dataset,
                sampler=data_sampler,
                batch_size=self._cfgs.batch_size,
                collate_fn=numpy_collate,
                drop_last=True,
                num_workers=8,
            )
        )

        if self._cfgs.algo_cfg.target_entropy >= 0.0:
            action_space = self._eval_sampler.env.action_space
            self._cfgs.algo_cfg.target_entropy = -np.prod(action_space.shape).item()

        # setup policy
        self._policy = self._setup_policy()
        self._policy_dist = GaussianPolicy(
            self._action_dim, temperature=self._cfgs.policy_temp
        )

        # setup Q-function
        self._qf = self._setup_qf()
        self._vf = self._setup_vf()

        # setup agent
        self._agent = DiffusionQL(
            self._cfgs.algo_cfg, self._policy, self._qf, self._vf, self._policy_dist
        )

        # setup sampler policy
        sampler_policy = SamplerPolicy(self._agent.policy, self._agent.qf)
        self._evaluator = self._setup_evaluator(sampler_policy, eval_sampler, dataset)

    def _setup_qf(self):
        qf = Critic(
            self._observation_dim,
            self._action_dim,
            to_arch(self._cfgs.qf_arch),
            use_layer_norm=self._cfgs.qf_layer_norm,
            act=self._act_fn,
            orthogonal_init=self._cfgs.orthogonal_init,
        )
        return qf

    def _setup_vf(self):
        vf = Value(
            self._observation_dim,
            to_arch(self._cfgs.qf_arch),
            use_layer_norm=self._cfgs.qf_layer_norm,
            act=self._act_fn,
            orthogonal_init=self._cfgs.orthogonal_init,
        )
        return vf

    def _setup_policy(self):
        gd = GaussianDiffusion(
            num_timesteps=self._cfgs.algo_cfg.num_timesteps,
            schedule_name=self._cfgs.algo_cfg.schedule_name,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            # min_value=-self._max_action,
            # max_value=self._max_action,
            sample_temperature=self._cfgs.algo_cfg.sample_temperature,
        )
        policy = DiffusionPolicy(
            diffusion=gd,
            observation_dim=self._observation_dim,
            action_dim=self._action_dim,
            arch=to_arch(self._cfgs.policy_arch),
            time_embed_size=self._cfgs.algo_cfg.time_embed_size,
            use_layer_norm=self._cfgs.policy_layer_norm,
            sample_method=self._cfgs.sample_method,
            dpm_steps=self._cfgs.algo_cfg.dpm_steps,
            dpm_t_end=self._cfgs.algo_cfg.dpm_t_end,
        )

        return policy
