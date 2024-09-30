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

from enum import IntEnum


class ENV(IntEnum):
    Mujoco = 3


ENV_MAP = {
    "hopper": ENV.Mujoco,
    "walker": ENV.Mujoco,
    "cheetah": ENV.Mujoco,
    "finger": ENV.Mujoco,
    "humanoid": ENV.Mujoco,
    "cartpole": ENV.Mujoco,
    "fish": ENV.Mujoco,
    "Ant": ENV.Mujoco,
    "Cheetah": ENV.Mujoco,
    "Hopper": ENV.Mujoco,
    "Walker": ENV.Mujoco,
    "Swimmer": ENV.Mujoco,
    "Point": ENV.Mujoco,
    "maze2d": ENV.Mujoco,
    "antmaze": ENV.Mujoco,
    "wind-maze" : ENV.Mujoco,
}

ENVNAME_MAP = {
    ENV.Mujoco: "Mujoco",
}


class DATASET(IntEnum):
    D4RL = 1


DATASET_MAP = {"d4rl": DATASET.D4RL}
DATASET_ABBR_MAP = {"d4rl": "D4RL"}
