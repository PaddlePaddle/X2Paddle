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


class Pad(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, input, pad):
        shape = input.shape
        dim = len(shape)
        if len(pad) == 2:
            data_format = "NCL"
        elif len(pad) == 4:
            data_format = "NCHW"
        elif len(pad) == 6:
            data_format = "NCDHW"
        if dim == 3 and len(pad) == 4:
            input = paddle.unsqueeze(input, [0])
            output = paddle.nn.functional.pad(input,
                                              pad,
                                              data_format=data_format)
            output = paddle.squeeze(output, [0])
        elif dim == 4 and len(pad) == 6:
            input = paddle.unsqueeze(input, [0])
            output = paddle.nn.functional.pad(input,
                                              pad,
                                              data_format=data_format)
            output = paddle.squeeze(output, [0])
        else:
            output = paddle.nn.functional.pad(input,
                                              pad,
                                              data_format=data_format)
        return output
