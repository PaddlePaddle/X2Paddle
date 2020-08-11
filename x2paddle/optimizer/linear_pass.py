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
from x2paddle.core.util import *
from x2paddle.core.paddle_graph import PaddleLayer, PaddleGraph
from x2paddle.optimizer.passes import Pass, Matcher, PyTorchMatcher


class LinearPass(Pass):
    def __init__(self):
        self.linear_index = 0
        super(LinearPass, self).__init__()

    def build_pattern(self):
        """ 构造fc层的模式。
        fc层模式python实现代码示例:
            x149 = 2
            x151 = x146.shape
            x151 = len(x151)
            x152 = x151 == x149
            if x152 :
                x147 = self.x147
                x154 = fluid.layers.transpose(x=x147, perm=[1, 0])
                x148 = self.x148
                x155 = fluid.layers.addmm(input=x148, x=x146, y=x154, beta=1, alpha=1)
                x153 = x155
            else:
                x147 = self.x147
                x157 = fluid.layers.transpose(x=x147, perm=[1, 0])
                x158 = fluid.layers.matmul(x=x146, y=x157)
                x159 = True
                if x159 :
                    x148 = self.x148
                    x161 = x158 + 1 * x148
                    x160 = x161
                else:
                    x160 = x158
                x153 = x160
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(0)], value=2)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(1)], value=1)
        self.pattern.add_layer(
            "prim.shape", inputs={'input': "fc-input-0"},
            outputs=[gen_name(2)])
        self.pattern.add_layer(
            "prim.len", inputs={'input': gen_name(2)}, outputs=[gen_name(2)])
        self.pattern.add_layer(
            "prim.eq",
            inputs={"eq0": gen_name(2),
                    "eq1": gen_name(0)},
            outputs=[gen_name(3)])
        self.pattern.add_layer("prim.if", {'input': gen_name(3)}, [gen_name(4)])
        self.pattern.outputs.append(gen_name(4))
        if_layer_a = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(if_layer_a)
        pattern_block0.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(5)],
            value="params[{}]".format(string(gen_name(5))))
        pattern_block0.add_layer(
            "fluid.layers.transpose",
            inputs={"x": gen_name(5)},
            outputs=[gen_name(6)],
            perm=[1, 0])
        pattern_block0.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(7)],
            value="params[{}]".format(string(gen_name(7))))
        pattern_block0.add_layer(
            "fluid.layers.addmm",
            inputs={"input": gen_name(7),
                    "x": "fc-input-0",
                    "y": gen_name(6)},
            outputs=[gen_name(8)],
            beta=1,
            alpha=1)
        if_layer_a.inputs["input-0"] = "fc-input-0"
        self.pattern.inputs.append("fc-input-0")
        pattern_block0.add_layer(
            "prim.equal", inputs={'input': gen_name(8)}, outputs=[gen_name(4)])
        if_layer_a.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(if_layer_a)
        pattern_block1.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(5)],
            value="params[{}]".format(string(gen_name(5))))
        pattern_block1.add_layer(
            "fluid.layers.transpose",
            inputs={"x": gen_name(5)},
            outputs=[gen_name(6)],
            perm=[1, 0])
        pattern_block1.add_layer(
            "fluid.layers.matmul",
            inputs={"x": "fc-input-0",
                    "y": gen_name(6)},
            outputs=[gen_name(9)])
        if_layer_a.inputs["input-1"] = "fc-input-0"
        pattern_block1.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(10)], value=True)
        pattern_block1.add_layer("prim.if", {'input': gen_name(10)},
                                 [gen_name(11)])
        if_layer_b = pattern_block1.layers[list(pattern_block1.layers.keys())[
            -1]]
        pattern_block1_block0 = PaddleGraph(if_layer_b)
        pattern_block1_block0.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(12)],
            value="params[{}]".format(string(gen_name(12))))
        pattern_block1_block0.add_layer(
            "prim.add",
            inputs={"x": gen_name(9),
                    "y": gen_name(12)},
            outputs=[gen_name(13)],
            alpha=1)
        if_layer_b.inputs["input-0"] = gen_name(9)
        pattern_block1_block0.add_layer(
            "prim.equal",
            inputs={'input': gen_name(13)},
            outputs=[gen_name(11)])
        if_layer_b.add_block(pattern_block1_block0)
        pattern_block1_block1 = PaddleGraph(if_layer_b)
        pattern_block1_block1.add_layer(
            "prim.equal", inputs={'input': gen_name(9)},
            outputs=[gen_name(11)])
        if_layer_b.inputs["input-1"] = gen_name(9)
        pattern_block1.add_layer(
            "prim.equal", inputs={'input': gen_name(11)},
            outputs=[gen_name(4)])
        if_layer_b.add_block(pattern_block1_block1)
        if_layer_a.add_block(pattern_block1)
        self.pattern.build(
            inputs={"input-0": "fc-input-0",
                    "input-1": "fc-input-0"})


class LinearMatcher(PyTorchMatcher):
    def __init__(self):
        self.linear_index = 0
        super(LinearMatcher, self).__init__()

    def replace_layer(self, graph, subgraph_global_layers):
        subgraph_global_layers_id = list(subgraph_global_layers.keys())
        layer = subgraph_global_layers[subgraph_global_layers_id[2]]
        input_name = layer.inputs["input"]
        layer = subgraph_global_layers[subgraph_global_layers_id[5]]
        output_name = layer.outputs[0]
        layer = subgraph_global_layers[subgraph_global_layers_id[6]]
        weight_name = layer.attrs["value"][8:-2]
        layer = subgraph_global_layers[subgraph_global_layers_id[8]]
        bias_name = layer.attrs["value"][8:-2]
        attrs = {}
        attrs["input_dim"] = graph.parameters[weight_name].shape[1]
        attrs["output_dim"] = graph.parameters[weight_name].shape[0]
        linear_name = "linear{}".format(self.linear_index)
        self.linear_index += 1
        graph.parameters["{}.weight".format(linear_name)] = graph.parameters[
            weight_name].transpose((1, 0))
        graph.parameters["{}.bias".format(linear_name)] = np.squeeze(
            graph.parameters[bias_name])
        graph.parameters.pop(weight_name)
        graph.parameters.pop(bias_name)
        for i, layer_id in enumerate(subgraph_global_layers):
            if layer_id in graph.layers:
                layer = graph.layers[layer_id]
                if i == 0:
                    new_layer = PaddleLayer(
                        layer_id,
                        "fluid.dygraph.Linear",
                        inputs={"input": input_name},
                        outputs=[linear_name, output_name],
                        **attrs)
                    graph.layers[layer_id] = new_layer
                else:
                    graph.layers.pop(layer_id)
        graph.build()
        return graph
