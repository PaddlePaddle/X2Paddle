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


def add_parambase_function(func):
    setattr(paddle.fluid.framework.ParamBase, func.__name__, func)


@add_parambase_function
def normal_(self, mean=0.0, std=1.0):
    replaced_param = paddle.create_parameter(
        shape=self.shape,
        dtype=self.dtype,
        default_initializer=paddle.nn.initializer.Normal(
            mean=mean, std=std))
    paddle.assign(self, replaced_param)


@add_parambase_function
def zero_(self):
    replaced_param = paddle.create_parameter(
        shape=self.shape,
        dtype=self.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.zeros(self.shape, self.dtype)))
    paddle.assign(self, replaced_param)


@add_parambase_function
def fill_(self, value):
    replaced_param = paddle.create_parameter(
        shape=self.shape,
        dtype=self.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.full(self.shape, value, self.dtype)))
    paddle.assign(self, replaced_param)
