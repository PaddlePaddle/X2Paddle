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
from paddle import _C_ops
from paddle import in_dynamic_mode
from paddle.common_ops_import import Variable, LayerHelper, check_variable_and_dtype, check_type, check_dtype


@paddle.jit.not_to_static
def roi_pool(input,
             rois,
             pooled_height,
             pooled_width,
             spatial_scale=1.0,
             rois_num=None,
             name=None):
    if in_dynamic_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        pool_out, argmaxes = _C_ops.roi_pool(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale)
        return pool_out, argmaxes

    else:
        check_variable_and_dtype(input, 'input', ['float32'], 'roi_pool')
        check_variable_and_dtype(rois, 'rois', ['float32'], 'roi_pool')
        helper = LayerHelper('roi_pool', **locals())
        dtype = helper.input_dtype()
        pool_out = helper.create_variable_for_type_inference(dtype)
        argmaxes = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {
            "X": input,
            "ROIs": rois,
        }
        if rois_num is not None:
            inputs['RoisNum'] = rois_num
        helper.append_op(
            type="roi_pool",
            inputs=inputs,
            outputs={"Out": pool_out,
                     "Argmax": argmaxes},
            attrs={
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "spatial_scale": spatial_scale
            })
        return pool_out, argmaxes


class ROIPooling(object):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.roipooling_layer_attrs = {
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale
        }

    def __call__(self, x0, x1):
        out = roi_pool(input=x0, rois=x1, **self.roipooling_layer_attrs)
        return out
