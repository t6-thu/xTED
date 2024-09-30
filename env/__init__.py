"""Codes are borrowed from tianshou (https://github.com/thu-ml/tianshou.git)"""
"""Env package."""

from env.gym_wrappers import (
    ContinuousToDiscrete,
    MultiDiscreteToDiscrete,
    TruncatedAsTerminated,
)
from env.venv_wrappers import VectorEnvNormObs, VectorEnvWrapper
from env.venvs import (
    BaseVectorEnv,
    DummyVectorEnv,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
    "ShmemVectorEnv",
    "RayVectorEnv",
    "VectorEnvWrapper",
    "VectorEnvNormObs",
    "ContinuousToDiscrete",
    "MultiDiscreteToDiscrete",
    "TruncatedAsTerminated",
]


def get_envs(env_fn, num_envs: int):
    if num_envs == 1:
        print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.")
        envs = DummyVectorEnv([env_fn])
    else:
        envs = SubprocVectorEnv([env_fn for _ in range(num_envs)])
    return envs
