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

"""General utils for training."""

import functools
import importlib
import os
import pprint
import random
import string
import sys
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import numpy as np
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags

from utilities.jax_utils import init_rng
from datetime import datetime
import gym

def to_arch(string):
    return tuple(int(x) for x in string.split("-"))


def apply_conditioning(x, conditions, condition_dim):
    import ipdb
    # ipdb.set_trace()
    for t, val in conditions.items():
        assert condition_dim is not None
        if type(val) is not tuple:
            try:
                # import ipdb
                # ipdb.set_trace()
                x = x.at[:, t, :condition_dim[0]].set(val[:condition_dim]) if type(condition_dim) is tuple else x.at[:, t, :condition_dim].set(val[:condition_dim])
            except:
                x = x.at[:, t, :val.shape[0]].set(val)
        else:
            obs, act, reward = val
            obs_dim, act_dim, reward_dim = condition_dim
            obs_dim -= act_dim+reward_dim
            x = x.at[:, t, :obs_dim].set(obs)
            x = x.at[:, t, obs_dim:act_dim+obs_dim].set(act)
            x = x.at[:,t, act_dim+obs_dim:].set(reward)
    return x


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def dot_key_dict_to_nested_dicts(dict_in):
    dict_out = {}
    for key, value in dict_in.items():
        cur = dict_out
        *keys, leaf = key.split(".")
        for k in keys:
            cur = cur.setdefault(k, {})
        cur[leaf] = value
    return dict_out


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None, run_name=''):
        config = ConfigDict()
        # config.team = "jax_diffrl"
        config.entity = 'ml_cat'
        config.online = True
        config.project = "jaxDiffusionRL"
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config.output_dir = f"logs/{now}_{run_name}"
        config.random_delay = 0.0
        config.log_dir = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
            
        config.output_dir = f"logs/{now}_{run_name}"
        return config

    def __init__(self, config, variant, run_name):
        self.config = self.get_default_config(updates = config, run_name = run_name)
        # import ipdb
        # ipdb.set_trace()
        if self.config.log_dir is None:
            self.config.log_dir = uuid.uuid4().hex

        if self.config.output_dir == "":
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(
                self.config.output_dir, self.config.log_dir
            )
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            entity=self.config.entity,
            reinit=True,
            config=self._variant,
            name=run_name,
            project=self.config.project,
            dir=self.config.output_dir,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode="online" if self.config.online else "offline",
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def print_flags(flags, flags_def):
    logging.info(
        "Running training with hyperparameters: \n{}".format(
            pprint.pformat(
                [
                    "{}: {}".format(key, val)
                    for key, val in get_user_flags(flags, flags_def).items()
                ]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


def import_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DotFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        return (self.get_value(field_name, args, kwargs), field_name)
    

# generate xml assets path: gym_xml_path
def generate_xml_path() -> str:
    import os
    import gym
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)
    print("gym_xml_path: ", xml_path)

    return xml_path


gym_xml_path = generate_xml_path()

def get_new_gravity_env(variety, env_name):
    update_target_env_gravity(variety, env_name)
    
    # import ipdb
    # ipdb.set_trace()
    env = gym.make(env_name)

    return env

def get_new_friction_env(variety, env_name):
    update_target_env_friction(variety, env_name)
    
    # import ipdb
    # ipdb.set_trace()
    env = gym.make(env_name)

    return env

def get_new_thigh_env(variety_degree, env_name):
    update_target_env_short_thigh(variety_degree, env_name)
    env = gym.make(env_name)
    return env

def get_new_torso_env(variety_degree, env_name):
    update_target_env_torso_length(variety_degree, env_name)
    env = gym.make(env_name)
    return env

def get_new_wind_env(variety, env_name):
    env = gym.make(env_name)
    env.model.opt.wind[:] = np.array([-variety, 0., 0.])
    
    return env

def update_target_env_gravity(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_gravityx{}.xml".format(old_xml_name.split(".")[0], variety_degree)
    import re
    with open('./h_2_o/xml_path/source_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('./h_2_o/xml_path/target_file/{}'.format(xml_name), "w+")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                gravity_list = a[0].split(" ")
                new_gravity_list = []
                for num in gravity_list:
                    new_gravity_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_gravity_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ./h_2_o/xml_path/target_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)

#TODO: density
def update_target_env_density(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./h_2_o/xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('../xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ../xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)

#TODO: friction
def update_target_env_friction(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_frictionx{}.xml".format(old_xml_name.split(".")[0], variety_degree)
    import re
    # import ipdb
    # ipdb.set_trace()
    with open('./h_2_o/xml_path/source_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('./h_2_o/xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./h_2_o/xml_path/target_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)
    
def update_target_env_short_thigh(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml
    xml_name = "{}_thigh_sizex{}.xml".format(old_xml_name.split(".")[0], variety_degree)
    import re
    with open('./h_2_o/xml_path/source_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('./h_2_o/xml_path/target_file/{}'.format(xml_name), "w+")
        for line in f.readlines():
            if "size" in line and "thigh" in line:
                pattern = re.compile(r"size=\"(.*?)\"")
                a = pattern.findall(line)
                thigh_range_list = a[0].split(" ")
                new_thigh_range_list = []
                for num in thigh_range_list:
                    new_thigh_range_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_thigh_range_list)
                replace_num = "size=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ./h_2_o/xml_path/target_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)

def update_target_env_torso_length(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml
    xml_name = "{}_torso_lengthx{}.xml".format(old_xml_name.split(".")[0], variety_degree)
    import re
    with open('./h_2_o/xml_path/source_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('./h_2_o/xml_path/target_file/{}'.format(xml_name), "w+")
        for line in f.readlines():
            if "fromto" in line and "torso" in line:
                pattern = re.compile(r"fromto=\"(.*?)\"")
                a = pattern.findall(line)
                thigh_range_list = a[0].split(" ")
                new_thigh_range_list = []
                for num in thigh_range_list:
                    new_thigh_range_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_thigh_range_list)
                replace_num = "fromto=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ./h_2_o/xml_path/target_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)




def parse_xml_name(env_name):
    # import ipdb
    # ipdb.set_trace()
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    elif "maze" in env_name.lower():
        xml_name = "point.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name



    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        assert torch.isnan(observations).sum() == 0, print(observations)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        assert torch.isnan(mean).sum() == 0, print(mean)
        assert torch.isnan(log_std).sum() == 0, print(log_std)
        return self.tanh_gaussian(mean, log_std, deterministic)




