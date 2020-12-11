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

def roipooling(x0,
               x1, 
               pooled_height, 
               pooled_width, 
               spatial_scale):
    roipooling_layer_attrs = {
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale}
    slice_x1 = paddle.slice(input=x1, axes=[1], 
                            starts=[1], ends=[5])
    out = fluid.layers.roi_pool(input=x0, 
                                rois=slice_x1, 
                                **roipooling_layer_attrs)
    return out
