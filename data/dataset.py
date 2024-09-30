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

"""Utils to generate trajectory based dataset with multi-step reward."""

from typing import List


class Dataset:
    __initialized = False

    def __init__(
        self,
        required_keys: List[str] = [
            "observations",
            "actions",
            "rewards",
            "terminals",
        ],
        verbose: bool = True,
        **kwargs,
    ):
        self._dict = {}
        for k, v in kwargs.items():
            self[k] = v

        self.required_keys = []
        self.extra_keys = []
        for k in self.keys():
            if k in required_keys:
                self.required_keys.append(k)
            else:
                self.extra_keys.append(k)
        assert set(self.required_keys) == set(
            required_keys
        ), f"Missing keys: {set(required_keys) - set(self.required_keys)}"
        if verbose:
            print("[ data/dataset.py ] Dataset: get required keys:", self.required_keys)
            print("[ data/dataset.py ] Dataset: get extra keys:", self.extra_keys)
        self.__initialized = True

    def __setattr__(self, k, v):
        if self.__initialized and k not in self._dict.keys():
            raise AttributeError(f"Cannot add new attributes to Dataset: {k}")
        else:
            object.__setattr__(self, k, v)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return {k: v for k, v in self._dict.items() if k != "traj_lengths"}.items()

    def __len__(self):
        return self._dict["observations"].shape[0]

    def __repr__(self):
        return "[ data/dataset.py ] Dataset:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        setattr(self, key, val)
