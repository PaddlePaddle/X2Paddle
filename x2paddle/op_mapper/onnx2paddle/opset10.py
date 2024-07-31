# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from .opset9 import OpSet9
from x2paddle.core.util import *
import numpy as np
import math


def print_mapping_info(func):

    def run_mapping(*args, **kwargs):
        node = args[1]
        try:
            res = func(*args, **kwargs)
        except:
            raise Exception("convert failed node:{}, op_type is {}".format(
                node.name[9:], node.layer_type))
        else:
            return res

    return run_mapping


def _get_same_padding(in_size, kernel_size, stride, autopad):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0
    if autopad == "SAME_UPPER":
        return [pad0, pad1]
    if autopad == "SAME_LOWER":
        return [pad1, pad0]


class OpSet10(OpSet9):

    def __init__(self, decoder, paddle_graph):
        super(OpSet10, self).__init__(decoder, paddle_graph)

    @print_mapping_info
    def AveragePool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)

        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        kernel_shape = node.get_attr("kernel_shape")
        count_include_pad = node.get_attr("count_include_pad", 0)
        # Support ceil_mode Since opset version >= 10
        ceil_mode = bool(node.get_attr("ceil_mode", 0))
        exclusive = True
        if count_include_pad > 0:
            exclusive = False
        poolnd = len(kernel_shape)
        strides = node.get_attr("strides")
        pad_mode = node.get_attr("pads")
        pads = node.get_attr('pads', [0] * (poolnd * 2))

        input_shape = val_x.out_shapes[0]
        paddings = np.array(pads).reshape((2, -1)).transpose().astype("int32")
        paddings = paddings.flatten().tolist()

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            # Warning: SAME_UPPER and SAME_LOWER does not yet support dynamic shapes
            if input_shape[2] == -1 or input_shape[3] == -1:
                _logger.warning(
                    'SAME_UPPER and SAME_LOWER does not yet support dynamic shapes, the conversion result may have a diff!!!'
                )
            pad_h = _get_same_padding(input_shape[2], kernel_shape[0],
                                      strides[0], auto_pad)
            pad_w = _get_same_padding(input_shape[3], kernel_shape[1],
                                      strides[1], auto_pad)
            paddings = pad_h + pad_w

        op_name = name_generator("pool", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        paddle_op = 'paddle.nn.AvgPool{}D'.format(poolnd)
        assert 1 <= poolnd <= 3, 'only Pool1D, Pool2D and Pool3D are supported'
        layer_attrs = {
            "kernel_size": kernel_shape,
            "stride": strides,
            "ceil_mode": ceil_mode,
            "padding": paddings,
            "exclusive": exclusive,
        }
        self.paddle_graph.add_layer(
            paddle_op,
            inputs={'x': val_x if isinstance(val_x, str) else val_x.name},
            outputs=layer_outputs,
            **layer_attrs)

    @print_mapping_info
    def Mod(self, node):
        # Support Mod op Since opset version >= 10
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        fmod = node.get_attr('fmod', 0)
        x_dtype = str(val_x.dtype)
        y_dtype = str(val_y.dtype)
        if "int" in x_dtype and "int" in y_dtype and fmod == 1:
            fmod = 0
        if fmod == 0:
            self.paddle_graph.add_layer('paddle.mod',
                                        inputs={
                                            "x": val_x.name,
                                            "y": val_y.name
                                        },
                                        outputs=[node.name])
        else:
            # Step1:trunc_div(a, b) = sign(a / b) * floor(abs(a / b))
            self.paddle_graph.add_layer('paddle.divide',
                                        inputs={
                                            "x": val_x.name,
                                            "y": val_y.name
                                        },
                                        outputs=[node.name + "_divide"])
            self.paddle_graph.add_layer('paddle.sign',
                                        inputs={"x": node.name + "_divide"},
                                        outputs=[node.name + "_sign"])
            self.paddle_graph.add_layer('paddle.abs',
                                        inputs={"x": node.name + "_divide"},
                                        outputs=[node.name + "_abs"])
            self.paddle_graph.add_layer('paddle.floor',
                                        inputs={"x": node.name + "_abs"},
                                        outputs=[node.name + "_floor"])
            self.paddle_graph.add_layer('paddle.multiply',
                                        inputs={
                                            "x": node.name + "_sign",
                                            "y": node.name + "_floor"
                                        },
                                        outputs=[node.name + "_trunc_div"])
            # Step2:result = a - trunc_div(a, b) * b
            self.paddle_graph.add_layer('paddle.multiply',
                                        inputs={
                                            "x": node.name + "_trunc_div",
                                            "y": val_y.name
                                        },
                                        outputs=[node.name + "_trunc_div"])
            self.paddle_graph.add_layer('paddle.subtract',
                                        inputs={
                                            "x": val_x.name,
                                            "y": node.name + "_trunc_div"
                                        },
                                        outputs=[node.name])

    @print_mapping_info
    def IsInf(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        if node.get_attr('detect_negative') != None or node.get_attr(
                'detect_positive') != None:
            if node.get_attr('detect_negative') != 1 or node.get_attr(
                    'detect_positive') != 1:
                raise Exception(
                    "x2addle does not currently support IsINF with attributes 'detect_negative' and 'detect_positive'."
                )
        else:
            self.paddle_graph.add_layer('paddle.isinf',
                                        inputs={"x": val_x.name},
                                        outputs=[node.name])
