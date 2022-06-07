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
def prior_box(input,
              image,
              min_sizes,
              max_sizes=None,
              aspect_ratios=[1.],
              variance=[0.1, 0.1, 0.2, 0.2],
              flip=False,
              clip=False,
              steps=[0.0, 0.0],
              offset=0.5,
              min_max_aspect_ratios_order=False,
              name=None):
    helper = LayerHelper("prior_box", **locals())
    dtype = helper.input_dtype()
    check_variable_and_dtype(
        input, 'input', ['uint8', 'int8', 'float32', 'float64'], 'prior_box')

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if not _is_list_or_tuple_(min_sizes):
        min_sizes = [min_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not (_is_list_or_tuple_(steps) and len(steps) == 2):
        raise ValueError('steps should be a list or tuple ',
                         'with length 2, (step_width, step_height).')

    min_sizes = list(map(float, min_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    steps = list(map(float, steps))

    cur_max_sizes = None
    if max_sizes is not None and len(max_sizes) > 0 and max_sizes[0] > 0:
        if not _is_list_or_tuple_(max_sizes):
            max_sizes = [max_sizes]
        cur_max_sizes = max_sizes

    if in_dynamic_mode():
        attrs = ('min_sizes', min_sizes, 'aspect_ratios', aspect_ratios,
                 'variances', variance, 'flip', flip, 'clip', clip, 'step_w',
                 steps[0], 'step_h', steps[1], 'offset', offset,
                 'min_max_aspect_ratios_order', min_max_aspect_ratios_order)
        if cur_max_sizes is not None:
            attrs += ('max_sizes', cur_max_sizes)
        box, var = _C_ops.prior_box(input, image, *attrs)
        return box, var
    else:
        attrs = {
            'min_sizes': min_sizes,
            'aspect_ratios': aspect_ratios,
            'variances': variance,
            'flip': flip,
            'clip': clip,
            'step_w': steps[0],
            'step_h': steps[1],
            'offset': offset,
            'min_max_aspect_ratios_order': min_max_aspect_ratios_order
        }

        if cur_max_sizes is not None:
            attrs['max_sizes'] = cur_max_sizes

        box = helper.create_variable_for_type_inference(dtype)
        var = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="prior_box",
            inputs={"Input": input,
                    "Image": image},
            outputs={"Boxes": box,
                     "Variances": var},
            attrs=attrs, )
        box.stop_gradient = True
        var.stop_gradient = True
        return box, var


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
        box, var = prior_box(input=x0, image=x1, **self.priorbox_layer_attrs)
        box = paddle.reshape(x=box, shape=[1, 1, -1])
        var = paddle.reshape(x=var, shape=[1, 1, -1])
        out = paddle.concat(x=[box, var], axis=1)
        return out
