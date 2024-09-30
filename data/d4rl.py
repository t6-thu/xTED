from functools import partial
from typing import Callable

import d4rl  # noqa
import gym

from utilities.utils import compose

from .dataset import Dataset
from .preprocess import clip_actions, pad_trajs_to_dataset, split_to_trajs
import ipdb
import numpy as np
from viskit.logging import logger

def truncate_dict_values(data_dict, percentage):
    truncated_dict = {}
    for key, value in data_dict.items():
        length = len(value)
        truncate_length = int(length * percentage)
        truncated_dict[key] = value[:truncate_length]
    return truncated_dict

def get_dataset(
    cfgs,
    env,
    data_path,
    max_traj_length: int,
    termination_penalty: float = None,
    include_next_obs: bool = False,
    clip_to_eps: bool = False,  # disable action clip for debugging purpose
):
    # ipdb.set_trace()
    preprocess_fn = compose(
        partial(
            pad_trajs_to_dataset,
            max_traj_length=max_traj_length,
            termination_penalty=termination_penalty,
            include_next_obs=include_next_obs,
        ),
        split_to_trajs,
        partial(
            clip_actions,
            clip_to_eps=clip_to_eps,
        ),
    )
    return D4RLDataset(cfgs, env, data_path, preprocess_fn=preprocess_fn)


class D4RLDataset(Dataset):
    def __init__(self, cfgs, env: gym.Env, data_path: dict, preprocess_fn: Callable, **kwargs):
        import ipdb
        
        # self.raw_dataset = dataset = env.get_dataset()
        # ipdb.set_trace()
        dataset = np.load(data_path, allow_pickle = True)[0] if len(np.load(data_path, allow_pickle = True).shape) == 1 else np.load(data_path, allow_pickle = True)[()]
        
        self.raw_dataset = truncate_dict_values(dataset, cfgs.real_ratio)
        # ipdb.set_trace()
        self.env_data_length = len(dataset['observations'])
        data_dict = preprocess_fn(dataset)
        # model_path = f'./model/{cfgs.env}_train_dataset.pkl' 
        # data_dict = logger.copy_load_SDEdit_data(model_path, 'train_dataset', load = True)
        
        super().__init__(**data_dict, **kwargs)
        
        
class OnlineDataset(Dataset):
    def __init__(self, dataset : dict, **kwargs):
        self.raw_dataset = dataset
        self.max_size = 3000
        super().__init__(**self.raw_dataset, **kwargs)
