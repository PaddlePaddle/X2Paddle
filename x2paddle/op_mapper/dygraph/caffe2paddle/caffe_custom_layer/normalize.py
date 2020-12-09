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

class Normalize(object):
    def __init__(self, axis, param_name, param_shape):
        self.axis = axis
        self.param_name = param_name
        self.param_shape = param_shape
        
    def __call__(self, x):
        l2 = fluid.layers.prior_box(x=x, p=2, axis=1)
        attr = fluid.ParamAttr(name=self.param_name, trainable=False)
        param = paddle.nn.Layer.create_parameter(shape=self.param_shape,
                                                 attr=atr)
        out = paddle.multiply(x=l2, y=param, axis=self.axis)
        return out
        
    