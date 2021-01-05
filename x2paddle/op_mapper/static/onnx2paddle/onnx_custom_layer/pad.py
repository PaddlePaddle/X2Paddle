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

def custom_pad(self, x, pad, value, mode):
    layer_attrs = {}
    layer_attrs['mode'] = string(mode)
    layer_attrs['data_format'] = string('NCHW')
    layer_attrs['value'] = value
    pad = paddle.reshape(pad, shape=[2, -1])
    pad = paddle.transpose(pad, perm=[1, 0])
    pad = paddle.reverse(pad, axis=[0])
    pad = paddle.flatten(pad)
    out = paddle.nn.functional.pad(x=x, pad=pad, **self.layer_attrs)
    return out