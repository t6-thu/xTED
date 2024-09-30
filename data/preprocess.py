import numpy as np
import copy
from utilities.data_utils import atleast_nd
import ipdb
import random

def clip_actions(dataset, clip_to_eps: bool = True, eps: float = 1e-5):
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
    return dataset


def compute_returns(traj):
    episode_return = 0
    for _, _, rew, *_ in traj:
        episode_return += rew
    return episode_return


def split_to_trajs(dataset):
    
    # d_length = len(dataset['observations'])
    dones_float = np.zeros_like(dataset["rewards"])  # truncated and terminal
    # d_t = dataset['terminals'] if 'terminals' in dataset.keys() else dataset['timeouts']
    d_t_all = dataset['terminals'] + dataset['timeouts'] if 'timeouts' in dataset.keys() else dataset['terminals']
    if "next_observations" not in dataset.keys():
        dataset["next_observations"] = np.zeros_like(dataset["observations"])
        for i in range(len(dones_float) - 1):
            if not d_t_all[i]:
                dataset["next_observations"][i] = copy.deepcopy(dataset["observations"][i + 1])
        
    for i in range(len(dones_float) - 1):
        if (
            np.linalg.norm(
                dataset["observations"][i + 1] - dataset["next_observations"][i]
            )
            > 1e-6
            or d_t_all[i] == True
        ):
            # ipdb.set_trace()
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1

    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append(
            (
                dataset["observations"][i],
                dataset["actions"][i],
                dataset["rewards"][i],
                dones_float[i],
                dones_float[i],
                dataset["next_observations"][i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])
    # import ipdb
    # ipdb.set_trace()
    
    # trajs: 2000 * 999 list, every element is a tuple
    # import ipdb
    # ipdb.set_trace()
    # random.shuffle(trajs)
    # import ipdb
    
    # ipdb.set_trace()
    return trajs



def pad_trajs_to_dataset(
    trajs,
    max_traj_length: int,
    termination_penalty: float = None,
    include_next_obs: bool = False,
):
    
    n_trajs = len(trajs)

    dataset = {}
    
    obs_dim, act_dim = trajs[0][0][0].shape[0], trajs[0][0][1].shape[0]
    dataset["observations"] = np.zeros(
        (n_trajs, max_traj_length, obs_dim), dtype=np.float32
    )
    dataset["actions"] = np.zeros((n_trajs, max_traj_length, act_dim), dtype=np.float32)
    dataset["rewards"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["terminals"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["dones_float"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["traj_lengths"] = np.zeros((n_trajs,), dtype=np.int32)
    if include_next_obs:
        dataset["next_observations"] = np.zeros(
            (n_trajs, max_traj_length, obs_dim), dtype=np.float32
        )
    import ipdb
    
    for idx, traj in enumerate(trajs):
        traj_length = len(traj)
        dataset["traj_lengths"][idx] = traj_length
        # ipdb.set_trace()
        dataset["observations"][idx, :traj_length] = atleast_nd(
            np.stack([ts[0] for ts in traj], axis=0),
            n=2,
        )
        dataset["actions"][idx, :traj_length] = atleast_nd(
            np.stack([ts[1] for ts in traj], axis=0),
            n=2,
        )
        dataset["rewards"][idx, :traj_length] = np.stack([ts[2] for ts in traj], axis=0)
        dataset["dones_float"][idx, :traj_length] = np.stack(
            [ts[3] for ts in traj], axis=0
        )
        # ipdb.set_trace()
        dataset["terminals"][idx, :traj_length] = np.stack(
            [bool(ts[4]) for ts in traj], axis=0
        )
        if include_next_obs:
            dataset["next_observations"][idx, :traj_length] = atleast_nd(
                np.stack([ts[5] for ts in traj], axis=0),
                n=2,
            )
        if dataset["terminals"][idx].any() and termination_penalty is not None:
            # import ipdb
            # ipdb.set_trace()
            dataset["rewards"][idx, traj_length - 1] += termination_penalty
    import ipdb
    # ipdb.set_trace()
    return dataset
