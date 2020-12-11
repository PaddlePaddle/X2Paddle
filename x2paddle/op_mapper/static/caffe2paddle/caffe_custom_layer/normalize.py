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

def normalize(x, axis, param_name, param_shape, param_dtype):
    l2 = fluid.layers.prior_box(x=x, p=2, axis=1)
    param = paddle.static.nn.create_parameter(shape=param_shape,
                                              dtype=string(param_dtype),
                                              name=string(param_name))
    out = paddle.multiply(x=l2, y=param, axis=axis)
    return out