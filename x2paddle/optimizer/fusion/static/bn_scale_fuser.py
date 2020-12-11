#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from x2paddle.optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class Static_BNScaleFuser(FuseBase):
    def __init__(self):
        super(Static_BNScaleFuser, self).__init__(graph_type="dygraph")

    def build_pattern(self):
        """ 描述需要替换的batchnorm2d图结构。
        batchnorm2d层模式python实现代码示例:
            conv5_bn = fluid.layers.batch_norm(input=conv5, is_test=True, param_attr=None, bias_attr=None, moving_mean_name='conv5_bn_mean', moving_variance_name='conv5_bn_variance', epsilon=9.999999747378752e-06, name='conv5_bn')
        conv5_scale_scale = fluid.ParamAttr(name='conv5_scale_scale')
        conv5_scale_cparam1 = fluid.layers.create_parameter(attr=conv5_scale_scale, dtype=conv5_bn.dtype, shape=[256], name='conv5_scale_cparam1', is_bias=True, default_initializer=Constant(value=1.0))
        conv5_scale_mul = fluid.layers.elementwise_mul(x=conv5_bn, y=conv5_scale_cparam1, axis=1)
        conv5_scale_offset = fluid.ParamAttr(name='conv5_scale_offset')
        conv5_scale_cparam2 = fluid.layers.create_parameter(attr=conv5_scale_offset, dtype=conv5_bn.dtype, shape=[256], name='conv5_scale_cparam2', is_bias=True, default_initializer=Constant(value=1.0))
        conv5_scale = fluid.layers.elementwise_add(x=conv5_scale_mul, y=conv5_scale_cparam2, axis=1)
        """

        def gen_name(id):
            return "x" + str(id)
        
        self.pattern.add_layer(
            "fluid.layers.batch_norm",
            inputs={"input": "bn-input-0"},
            outputs=[gen_name(0)])
        self.pattern.add_layer(
            "fluid.ParamAttr",
            inputs={},
            outputs=[gen_name(1)])
        self.pattern.add_layer(
            "fluid.layers.create_parameter",
            inputs={"attr": gen_name(1)},
            outputs=[gen_name(2)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(0)
        inputs_dict['y'] = gen_name(2)
        self.pattern.add_layer(
            "fluid.layers.elementwise_mul",
            inputs=inputs_dict,
            outputs=[gen_name(3)])
        self.pattern.add_layer(
            "fluid.ParamAttr",
            inputs={},
            outputs=[gen_name(4)])
        self.pattern.add_layer(
            "fluid.layers.create_parameter",
            inputs={"attr": gen_name(4)},
            outputs=[gen_name(5)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(3)
        inputs_dict['y'] = gen_name(5)
        self.pattern.add_layer(
            "fluid.layers.elementwise_add",
            inputs=inputs_dict,
            outputs=[gen_name(6)])
        self.pattern.build(inputs={"input-0": "bn-input-0"})

    def insert_new_layer(self, graph, parameters, matches):
        new_layer = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layer
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers_id = list(matches.keys())
        layer = matches[layers_id[0]]
        layer_inputs = layer.inputs
        layer_name = layer.outputs[0]
        layer_attrs = layer.attrs
        layer_attrs["param_attr"] = string("{}_scale".format(layer_name))
        layer_attrs["bias_attr"] = string("{}_offset".format(layer_name))
        layer = matches[layers_id[-1]]
        layer_outputs = layer.outputs
        layer = matches[layers_id[1]]
        layer_name = layer.outputs[0]
        scale_numpy = parameters.pop(layer_name)
        parameters[layer_attrs["param_attr"][1: -1]] = scale_numpy
        layer = matches[layers_id[4]]
        layer_name = layer.outputs[0]
        scale_numpy = parameters.pop(layer_name)
        parameters[layer_attrs["bias_attr"][1: -1]] = scale_numpy
        new_layer = PaddleLayer(
            layers_id[0],
            "fluid.layers.batch_norm",
            inputs=layer_inputs,
            outputs=layer_outputs,
            **layer_attrs)
        return new_layer