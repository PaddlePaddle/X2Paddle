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


class PadAllDim4WithOneInput(object):
    def __init__(self, pad, value, mode):
        self.layer_attrs = {}
        self.layer_attrs['mode'] = mode
        self.layer_attrs['data_format'] = 'NCHW'
        self.layer_attrs['value'] = value
        self.pad1 = pad[0:4]
        self.pad2 = pad[4:9]

    def __call__(self, x):
        x = paddle.nn.functional.pad(x=x, pad=self.pad1, **self.layer_attrs)
        x = paddle.transpose(x, perm=[2, 3, 0, 1])
        x = paddle.nn.functional.pad(x=x, pad=self.pad2, **self.layer_attrs)
        out = paddle.transpose(x, perm=[2, 3, 0, 1])
        return out
