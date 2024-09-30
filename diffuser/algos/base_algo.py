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

import abc


class Algo(abc.ABC):
    """An Algo object corresponds to a learning agent which contains parameters and training steps."""

    @abc.abstractmethod
    def train(self, batch, **kwargs):
        """Train step of an agent."""

    @abc.abstractmethod
    def _train_step(self, train_state, target_params, rng, batch, **kwargs):
        """A single step for training."""

    @property
    @abc.abstractmethod
    def model_keys(self):
        """A tuple of learnable model keys."""

    @property
    @abc.abstractmethod
    def train_states(self):
        """The train states of this agent."""

    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    @abc.abstractmethod
    def total_steps(self):
        """Total training steps."""
