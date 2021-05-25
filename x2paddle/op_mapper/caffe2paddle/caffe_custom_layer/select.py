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


class Select(object):
    def __init__(self, input_shape, point, axis):
        self.point = point
        self.input_shape = input_shape
        self.axis = axis

    def __call__(self, x):
        start = self.point[0]
        if len(self.point) == 2:
            end = self.point[1]
        else:
            end = self.input_shape[self.axis]
        out = paddle.slice(x=x, start=start, end=end, axes=[self.axis])
        return out
