from collections import deque
from typing import Any, Dict

import numpy as np

from .base_evaluator import BaseEvaluator


class Generator(BaseEvaluator):
    eval_mode: str = "online"

    def __init__(self, config, policy, gen_sampler, diffuser):
        super().__init__(config, policy)
        self._gen_sampler = gen_sampler

        self._act_methods = self._cfgs.act_method.split("-")
        self._recent_returns = {
            method: deque(maxlen=10) for method in self._act_methods
        }
        self._best_returns = {method: -float("inf") for method in self._act_methods}
        self._diffuser = diffuser

    def generate(self, gen_n_trajs, joint_noise_mean = 0., joint_noise_std = 0.):
        for method in self._act_methods:
            trajs = self._sample_trajs(method, gen_n_trajs, joint_noise_mean, joint_noise_std)
        return trajs

    def update_policy(self, policy):
        self._policy = policy

    def _sample_trajs(self, act_method: str, gen_n_trajs : int, joint_noise_mean : float, joint_noise_std : float):
        self._policy.act_method = act_method
        trajs = self._gen_sampler.sample(
            self._policy,
            gen_n_trajs,
            deterministic=False,
            joint_noise_mean = joint_noise_mean,
            joint_noise_std=joint_noise_std,
        )
        return trajs

    def evaluate(self, epoch: int) -> Dict[str, Any]:
        pass
