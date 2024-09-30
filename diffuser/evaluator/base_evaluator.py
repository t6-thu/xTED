import csv
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from viskit.logging import logger, mkdir_p


class BaseEvaluator(ABC):
    def __init__(self, config, policy):
        self._cfgs = config
        self._policy = policy

    def dump_metrics(self, metrics: Dict[str, Any], epoch: int, suffix: str = ""):
        save_path = os.path.join(
            logger.get_snapshot_dir(), "results", f"metrics_{epoch}{suffix}.csv"
        )
        mkdir_p(os.path.dirname(save_path))
        with open(save_path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

    def update_params(self, params):
        self._policy.update_params(params)

    @abstractmethod
    def evaluate(self, epoch: int) -> Dict[str, Any]:
        pass
