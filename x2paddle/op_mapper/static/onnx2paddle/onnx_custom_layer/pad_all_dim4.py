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

def pad_all_dim4(x, pad, value, mode):
    pad = paddle.reshape(pad, shape=[2, -1])
    pad = paddle.transpose(pad, perm=[1, 0])
    pad = paddle.reverse(pad, axis=[0])
    pad = paddle.flatten(pad)
    pad = paddle.cast(pad, dtype="int32")        
    pad1, pad2 = paddle.split(pad, num_or_sections=2, axis=0)
    x = paddle.nn.functional.pad(x=x, 
                                 pad=pad1, 
                                 mode=mode,
                                 data_format='NCHW',
                                 value=value)
    x = paddle.transpose(x, perm=[2, 3, 0, 1])
    x = paddle.nn.functional.pad(x=x, 
                                 pad=pad2,
                                 mode=mode,
                                 data_format='NCHW',
                                 value=value)
    out = paddle.transpose(x, perm=[2, 3, 0, 1])
    return out