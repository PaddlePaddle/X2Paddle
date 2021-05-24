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


class PriorBox(object):
    def __init__(self, min_sizes, max_sizes, aspect_ratios, variance, flip,
                 clip, steps, offset, min_max_aspect_ratios_order):
        self.priorbox_layer_attrs = {
            "min_sizes": min_sizes,
            "max_sizes": max_sizes,
            "aspect_ratios": aspect_ratios,
            "variance": variance,
            "flip": flip,
            "clip": clip,
            "steps": steps,
            "offset": offset,
            "min_max_aspect_ratios_order": min_max_aspect_ratios_order
        }

    def __call__(self, x0, x1):
        box, var = fluid.layers.prior_box(
            input=x0, image=x1, **self.priorbox_layer_attrs)
        box = paddle.reshape(x=box, shape=[1, 1, -1])
        var = paddle.reshape(x=var, shape=[1, 1, -1])
        out = paddle.concat(x=[box, var], axis=1)
        return out
