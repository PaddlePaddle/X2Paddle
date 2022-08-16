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

from .opset10 import OpSet10

from x2paddle.decoder.onnx_decoder import ONNXGraphDataNode
from x2paddle.core.util import *


def _const_weight_or_none(node, necessary=False):
    if 'Constant' in node.layer_type:
        return node.value
    if isinstance(node, ONNXGraphDataNode):
        return node.weight
    if necessary:
        assert '{} should be an initializer or Constant operator.'.format(
            node.name)
    return None


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


class OpSet11(OpSet10):
    def __init__(self, decoder, paddle_graph):
        super(OpSet11, self).__init__(decoder, paddle_graph)

    @print_mapping_info
    def Clip(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        max_value, min_value = None, None

        if len(node.inputs) == 1:
            if val_x.dtype != 'float32' and val_x.dtype != 'int32' and val_x.dtype != 'float64':
                indices_x_cast = val_x.name + '_cast'
                self.paddle_graph.add_layer(
                    'paddle.cast',
                    inputs={"x": val_x.name},
                    outputs=[indices_x_cast],
                    dtype=string('float32'))
                self.paddle_graph.add_layer(
                    'paddle.clip',
                    inputs={"x": indices_x_cast, },
                    outputs=[node.name], )
                self.paddle_graph.add_layer(
                    'paddle.cast',
                    inputs={"x": node.name},
                    outputs=[node.name],
                    dtype=string(val_x.dtype))

            else:
                self.paddle_graph.add_layer(
                    'paddle.clip',
                    inputs={"x": val_x.name},
                    outputs=[node.name], )
        else:
            if len(node.inputs) == 2:
                val_ipt = self.graph.get_input_node(node, idx=1, copy=True)
                index = node.get_input_index(val_ipt.name)
                if val_x.dtype != 'float32' and val_x.dtype != 'int32' and val_x.dtype != 'float64':
                    indices_val_cast = val_ipt.name + '_cast'
                    indices_x_cast = val_x.name + '_cast'
                    self.paddle_graph.add_layer(
                        'paddle.cast',
                        inputs={"x": val_ipt.name},
                        outputs=[indices_val_cast],
                        dtype=string('float32'))
                    self.paddle_graph.add_layer(
                        'paddle.cast',
                        inputs={"x": val_x.name},
                        outputs=[indices_x_cast],
                        dtype=string('float32'))

                    if index == 1:
                        self.paddle_graph.add_layer(
                            'paddle.clip',
                            inputs={
                                "x": indices_x_cast,
                                "min": indices_val_cast
                            },
                            outputs=[node.name], )

                    if index == 2:
                        self.paddle_graph.add_layer(
                            'paddle.clip',
                            inputs={
                                "x": indices_x_cast,
                                "max": indices_val_cast
                            },
                            outputs=[node.name], )
                    self.paddle_graph.add_layer(
                        'paddle.cast',
                        inputs={"x": node.name},
                        outputs=[node.name],
                        dtype=string(val_x.dtype))
                else:
                    if index == 1:
                        self.paddle_graph.add_layer(
                            'paddle.clip',
                            inputs={"x": val_x.name,
                                    "min": val_ipt.name},
                            outputs=[node.name], )
                    else:
                        self.paddle_graph.add_layer(
                            'paddle.clip',
                            inputs={"x": val_x.name,
                                    "max": val_ipt.name},
                            outputs=[node.name], )
            else:
                if len(node.inputs) == 3:
                    min_ipt = self.graph.get_input_node(node, idx=1, copy=True)
                    max_ipt = self.graph.get_input_node(node, idx=2, copy=True)
                    if val_x.dtype != 'float32' and val_x.dtype != 'int32' and val_x.dtype != 'float64':
                        indices_min_cast = min_ipt.name + '_cast'
                        indices_max_cast = max_ipt.name + '_cast'
                        indices_x_cast = val_x.name + '_cast'
                        self.paddle_graph.add_layer(
                            'paddle.cast',
                            inputs={"x": min_ipt.name},
                            outputs=[indices_min_cast],
                            dtype=string('float32'))
                        self.paddle_graph.add_layer(
                            'paddle.cast',
                            inputs={"x": max_ipt.name},
                            outputs=[indices_max_cast],
                            dtype=string('float32'))
                        self.paddle_graph.add_layer(
                            'paddle.cast',
                            inputs={"x": val_x.name},
                            outputs=[indices_x_cast],
                            dtype=string('float32'))
                        self.paddle_graph.add_layer(
                            'paddle.clip',
                            inputs={
                                "x": indices_x_cast,
                                "min": indices_min_cast,
                                "max": indices_max_cast
                            },
                            outputs=[node.name])
                        self.paddle_graph.add_layer(
                            'paddle.cast',
                            inputs={"x": node.name},
                            outputs=[node.name],
                            dtype=string(val_x.dtype))
                    else:
                        self.paddle_graph.add_layer(
                            'paddle.clip',
                            inputs={
                                "x": val_x.name,
                                "min": min_ipt.name,
                                "max": max_ipt.name
                            },
                            outputs=[node.name])
                else:
                    raise Exception("max_value or min_value can't be None")
