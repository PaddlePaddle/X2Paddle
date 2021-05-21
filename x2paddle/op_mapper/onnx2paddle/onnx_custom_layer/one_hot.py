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


class OneHot(object):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, indices, depth, values):
        indices_shape = indices.shape
        rank = len(indices.shape)
        real_axis = self.axis
        if self.axis < 0:
            real_axis = self.axis + rank + 1
        depth_range = paddle.arange(end=depth)
        ls = tuple(indices_shape[0:real_axis])
        rs = tuple(indices_shape[real_axis:rank])
        targets = paddle.reshape(depth_range, (1, ) *
                                 (real_axis - 0) + tuple(depth_range.shape) +
                                 (1, ) * (rank - real_axis))
        mod = paddle.mod(indices, depth)
        v = paddle.reshape(mod, ls + (1, ) + rs)
        out = targets == v
        out = paddle.cast(out, "float32")
        on_value = paddle.slice(values, axes=[0], starts=[1], ends=[2])
        off_value = paddle.slice(values, axes=[0], starts=[0], ends=[1])
        out = out * (on_value - off_value) + off_value
        return out
