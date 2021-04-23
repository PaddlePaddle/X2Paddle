# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from functools import partial


def add_layer_function(func):
    setattr(paddle.nn.Layer, func.__name__, func)


@property
def module(self):
    if hasattr(self, "_layers"):
        return self._layers
    else:
        return self


setattr(paddle.nn.Layer, "module", module)


@add_layer_function
def load_state_dict(self, state_dict, strict=True):
    for key, param in self.state_dict().items():
        state = state_dict.get(key, None)
        if state is None:
            if key.endswith(".scale"):
                state_dict[key] = state_dict.pop(key[0:-5] + "weight")
    self.set_state_dict(state_dict)


@add_layer_function
def to(self, *args, **kwargs):
    # TODO(syf): for dtype
    return self


@add_layer_function
def cuda(self):
    return self


@add_layer_function
def apply(self, func):
    func(self)


@add_layer_function
def modules(self):
    return [self] + self.sublayers()


@add_layer_function
def add_module(self, name, module):
    self.add_sublayer(name, module)


pd_cuda = partial(paddle.nn.Layer.cuda)


@add_layer_function
def cuda(self, device=None):
    return self


pd_train = partial(paddle.nn.Layer.train)


@add_layer_function
def train(self, mode=True):
    if mode:
        return pd_train(self)
    else:
        return paddle.nn.Layer.eval(self)
