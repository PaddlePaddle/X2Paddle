# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from x2paddle.core.util import *


class Gather(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x, index):
        if self.dim < 0:
            self.dim += len(x.shape)
        x_range = list(range(len(x.shape)))
        x_range[0] = self.dim
        x_range[self.dim] = 0
        x_swaped = paddle.transpose(x, perm=x_range)
        index_range = list(range(len(index.shape)))
        index_range[0] = self.dim
        index_range[self.dim] = 0
        index_swaped = paddle.transpose(index, perm=index_range)
        dtype = index.dtype

        x_shape = paddle.shape(x_swaped)
        index_shape = paddle.shape(index_swaped)

        prod = paddle.cast(paddle.prod(x_shape), dtype=dtype) / x_shape[0]

        x_swaped_flattend = paddle.flatten(x_swaped)
        index_swaped_flattend = paddle.flatten(index_swaped)
        index_swaped_flattend *= prod

        bias = paddle.arange(start=0, end=prod, dtype=dtype)
        bias = paddle.reshape(bias, x_shape[1:])
        bias = paddle.crop(bias, index_shape[1:])
        bias = paddle.flatten(bias)
        bias = paddle.tile(bias, [index_shape[0]])
        index_swaped_flattend += bias

        gathered = paddle.index_select(x_swaped_flattend, index_swaped_flattend)
        gathered = paddle.reshape(gathered, index_swaped.shape)

        out = paddle.transpose(gathered, perm=x_range)

        return out
