#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from .register import register


def InstanceNormalization_shape(input_shape):
    return input_shape


def InstanceNormalization_layer(inputs, name=None):
    # TODO(lvmengsi@baidu.com): Check the accuracy when using fluid.layers.layer_norm.
    epsilon = 1e-5
    input_ = inputs[0]
    mean = fluid.layers.reduce_mean(input_, dim=[2, 3], keep_dim=True)
    var = fluid.layers.reduce_mean(fluid.layers.square(input_ - mean),
                                   dim=[2, 3],
                                   keep_dim=True)
    if name is not None:
        scale_name = name + "_scale"
        offset_name = name + "_offset"

    scale_param = inputs[1]
    offset_param = inputs[2]
    scale = fluid.layers.create_parameter(name=scale_param.name,
                                          shape=input_.shape[1:2],
                                          dtype="float32")
    offset = fluid.layers.create_parameter(name=offset_param.name,
                                           shape=input_.shape[1:2],
                                           dtype="float32")

    tmp = fluid.layers.elementwise_mul(x=(input_ - mean), y=scale, axis=1)
    tmp = tmp / fluid.layers.sqrt(var + epsilon)
    tmp = fluid.layers.elementwise_add(tmp, offset, axis=1)
    return tmp


def InstanceNormalization_weights(name, data=None):
    weights_name = [name + '_scale']
    return weights_name


register(kind='InstanceNormalization',
         shape=InstanceNormalization_shape,
         layer=InstanceNormalization_layer,
         child_func=None,
         weights=InstanceNormalization_weights)
