from typing import Any, Dict

from .base_evaluator import BaseEvaluator


class SkipEvaluator(BaseEvaluator):
    eval_mode: str = "skip"

    def dump_metrics(self, metrics: Dict[str, Any], epoch: int, suffix: str = ""):
        pass

    def update_params(self, params):
        pass

    def evaluate(self, epoch: int) -> Dict[str, Any]:
        return {}
