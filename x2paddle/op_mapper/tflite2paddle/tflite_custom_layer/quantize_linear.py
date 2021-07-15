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
import paddle.fluid as fluid
from tflite.TensorType import TensorType


class QuantizeLinear(object):
    def __init__(self, zero_point, scale, dtype):
        self.zero_point = zero_point
        self.scale = scale
        if dtype == TensorType.INT8:
            self.dtype = "int8"
        elif dtype == TensorType.UINT8:
            self.dtype = "uint8"
        else:
            raise Exception("QuantizeLinear only support int8/uint8.")

    def __call__(self, x):
        x = x / self.scale + self.zero_point
        min_t = paddle.zeros_like(x, dtype="float32")
        max_t = paddle.full_like(x, 255.0, dtype="float32")
        x = paddle.where(x > 0.0, x, min_t)
        x = paddle.where(x < 255.0, x, max_t)
        x = paddle.round(x)
        out = paddle.cast(x, self.dtype)
        return out
