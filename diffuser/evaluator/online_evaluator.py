from collections import deque
from typing import Any, Dict

import numpy as np

from .base_evaluator import BaseEvaluator


class OnlineEvaluator(BaseEvaluator):
    eval_mode: str = "online"

    def __init__(self, config, policy, eval_sampler):
        super().__init__(config, policy)
        self._eval_sampler = eval_sampler

        self._act_methods = self._cfgs.act_method.split("-")
        self._recent_returns = {
            method: deque(maxlen=10) for method in self._act_methods
        }
        self._best_returns = {method: -float("inf") for method in self._act_methods}

    def evaluate(self, epoch: int) -> Dict[str, Any]:
        metrics = {}
        for method in self._act_methods:
            # import ipdb
            # ipdb.set_trace()
            trajs = self._sample_trajs(method)

            post = "" if len(self._act_methods) == 1 else "_" + method
            metrics["average_return" + post] = np.mean(
                [np.sum(t["rewards"]) for t in trajs]
            )
            metrics["average_traj_length" + post] = np.mean(
                [len(t["rewards"]) for t in trajs]
            )
            metrics["average_normalizd_return" + post] = cur_return = np.mean(
                [
                    self._eval_sampler.env.get_normalized_score(np.sum(t["rewards"]))
                    for t in trajs
                ]
            )
            self._recent_returns[method].append(cur_return)
            metrics["average_10_normalized_return" + post] = np.mean(
                self._recent_returns[method]
            )
            metrics["best_normalized_return" + post] = self._best_returns[method] = max(
                self._best_returns[method], cur_return
            )
            metrics["done" + post] = np.mean([np.sum(t["dones"]) for t in trajs])

        self.dump_metrics(metrics, epoch, suffix="_online")
        return metrics

    def _sample_trajs(self, act_method: str):
        self._policy.act_method = act_method
        # import ipdb
        # ipdb.set_trace()
        trajs = self._eval_sampler.sample(
            self._policy,
            self._cfgs.eval_n_trajs,
            deterministic=False,
        )
        return trajs
