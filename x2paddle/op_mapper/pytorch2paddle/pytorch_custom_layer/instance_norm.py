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
from paddle.nn.functional import instance_norm
from paddle.fluid.initializer import Constant


class InstanceNorm(paddle.nn.Layer):
    """
    This class is based class for InstanceNorm1D, 2d, 3d.
    See InstaceNorm1D, InstanceNorm2D or InstanceNorm3D for more details.
    """

    def __init__(self,
                 num_features,
                 epsilon=1e-5,
                 momentum=0.9,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW",
                 name=None):
        super(InstanceNorm, self).__init__()

        if weight_attr == False or bias_attr == False:
            assert weight_attr == bias_attr, "weight_attr and bias_attr must be set to Fasle at the same time in InstanceNorm"
        self._epsilon = epsilon
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        if weight_attr != False and bias_attr != False:
            self.scale = self.create_parameter(
                attr=self._weight_attr,
                shape=[num_features],
                default_initializer=Constant(1.0),
                is_bias=False)
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[num_features],
                default_initializer=Constant(0.0),
                is_bias=True)
        else:
            self.scale = None
            self.bias = None

    def forward(self, input):
        return instance_norm(
            input, weight=self.scale, bias=self.bias, eps=self._epsilon)

    def extra_repr(self):
        return 'num_features={}, epsilon={}'.format(self.scale.shape[0],
                                                    self._epsilon)
