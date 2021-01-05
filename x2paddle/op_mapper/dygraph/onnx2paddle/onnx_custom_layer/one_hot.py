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
        indices_shape = paddle.shape(indices)
        tmp = paddle.ones_like(indices_shape, dtype="int32")
        rank = paddle.sum(tmp)
        depth_range = paddle.arange(end=depth)
        zero = paddle.zeros([1], dtype="int32")
        one = paddle.ones([1], dtype="int32")
        axis = self.axis * one
        new_axis = axis + rank + 1
        cond = paddle.less_than(axis, zero)
        real_axis = paddle.where(cond, new_axis, axis)
        ls = paddle.slice(indices_shape, axes=[0], starts=[0], ends=real_axis)
        rs = paddle.slice(indices_shape, axes=[0], starts=real_axis, ends=rank)
        tmp = paddle.ones_like(ls, dtype="int32")
        ls_len = paddle.sum(tmp)
        ls_list = paddle.ones(ls_len, dtype="int32")
        tmp = paddle.ones_like(rs, dtype="int32")
        rs_len = paddle.sum(tmp)
        rs_list = paddle.ones(rs_len, dtype="int32")
        depth_range_shape = paddle.shape(depth_range)
        targets_shape = paddle.concat([ls_list, depth_range_shape, rs_list], axis=0)
        targets = paddle.reshape(depth_range, targets_shape)
        mod = paddle.mod(indices, depth)
        v_shape = paddle.concat([ls, paddle.shape(one), rs], axis=0)
        v = paddle.reshape(mod, v_shape)
        out = targets == v
        out = paddle.cast(out, "float32")
        on_value = paddle.slice(values, axes=[0], starts=[1], ends=[2])
        off_value = paddle.slice(values, axes=[0], starts=[0], ends=[1])
        out = out * (on_value - off_value) + off_value
        return out