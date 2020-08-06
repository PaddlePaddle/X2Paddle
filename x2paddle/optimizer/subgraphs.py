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


class Fuser(object):
    def subgraph_matcher(self, graph, graph_start_index, subgraph):
        subgraph_index = 0
        match_index = []
        for i, layer in enumerate(graph.layers[graph_start_index:]):
            subgraph_layer = subgraph.layers[subgraph_index]
            if layer.kernel == subgraph_layer.kernel:
                match_index.append(graph_start_index + i)
                if layer.kernel == "prim.constant":
                    if layer.attrs["value"] != subgraph_layer.attrs["value"]:
                        return False
                elif layer.kernel == "fluid.layers.addmm":
                    if layer.attrs["beta"] != subgraph_layer.attrs["beta"]:
                        return False
                    if layer.attrs["alpha"] != subgraph_layer.attrs["alpha"]:
                        return False

                is_check_input = True
                if graph_start_index + i in graph.edges_in:
                    if subgraph_index not in subgraph.edges_in:
                        return False
                    else:
                        if len(graph.edges_in[graph_start_index + i]) != len(
                                subgraph.edges_in[subgraph_index]):
                            return False
                else:

                    is_check_input = False
                if is_check_input:
                    layer_in = graph.edges_in[graph_start_index + i]
                    subgraph_layer_in = subgraph.edges_in[subgraph_index]
                    for j in range(len(layer_in)):
                        if subgraph_layer_in[j] != -1 and \
                        layer_in[j] != match_index[subgraph_layer_in[j]]:
                            if isinstance(match_index[subgraph_layer_in[j]], dict) and \
                            layer_in[j] == list(match_index[subgraph_layer_in[j]].keys())[0]:
                                continue
                            return False

                if graph_start_index + i in graph.edges_out:
                    if subgraph_index not in subgraph.edges_out:
                        if not set(subgraph_layer.outputs).issubset(
                                subgraph.outputs):
                            return False
                    else:
                        if len(graph.edges_out[graph_start_index + i]) != len(
                                subgraph.edges_out[subgraph_index]):
                            # 如果在每个节点edges_in相同的情况下，edges_out数目相同则说明无节点在subgraph外被用到
                            if not set(subgraph_layer.outputs).issubset(
                                    subgraph.outputs):
                                # 如若是输出节点不需要看输出edges
                                return False

                if layer.kernel == "prim.if":
                    res_list = []
                    last_index = match_index[-1]
                    res = self.subgraph_matcher(layer.blocks[0], 0,
                                                subgraph_layer.blocks[0])
                    if res:
                        res_list.append(res)
                        match_index[-1] = {last_index: res_list}
                    else:
                        return False
                    res = self.subgraph_matcher(layer.blocks[1], 0,
                                                subgraph_layer.blocks[1])
                    if res:
                        res_list.append(res)
                        match_index[-1] = {last_index: res_list}
                    else:
                        return False
                subgraph_index += 1
                if subgraph_index == len(subgraph.layers):
                    return match_index
        return False

    def run(self, graph):
        while True:
            is_stop = False
            for i, layer in enumerate(graph.layers):
                is_match = self.subgraph_matcher(graph, i, self.subgraph)
                if is_match:
                    graph = self.replace_layer(graph, is_match)
                    break
                for j, block in enumerate(layer.blocks):
                    if len(block.layers) > 0:
                        layer.blocks[j] = self.run(block)
                if i + 1 == len(graph.layers):
                    return graph
            if is_stop:
                break
        return graph


class FCFuser(Fuser):
    def __init__(self):
        self.subgraph = PaddleGraph()
        self.linear_index = 0
        self.build_subgraph()

    def build_subgraph(self):
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

        def gen_name(index):
            return "x" + str(index)

        self.subgraph.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(0)], value=2)
        self.subgraph.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(1)], value=1)
        self.subgraph.add_layer(
            "prim.shape", inputs={'input': "fc-input-0"},
            outputs=[gen_name(2)])
        self.subgraph.add_layer(
            "prim.len", inputs={'input': gen_name(2)}, outputs=[gen_name(2)])
        self.subgraph.add_layer(
            "prim.eq",
            inputs={"eq0": gen_name(2),
                    "eq1": gen_name(0)},
            outputs=[gen_name(3)])
        self.subgraph.add_layer("prim.if", {'input': gen_name(3)},
                                [gen_name(4)])
        self.subgraph.outputs.append(gen_name(4))
        if_layer_a = self.subgraph.layers[-1]
        subgraph_block0 = PaddleGraph()
        subgraph_block0.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(5)],
            value="middle_numpy[{}]".format(string(gen_name(5))))
        subgraph_block0.add_layer(
            "fluid.layers.transpose",
            inputs={"x": gen_name(5)},
            outputs=[gen_name(6)],
            perm=[1, 0])
        subgraph_block0.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(7)],
            value="middle_numpy[{}]".format(string(gen_name(7))))
        subgraph_block0.add_layer(
            "fluid.layers.addmm",
            inputs={"input": gen_name(7),
                    "x": "fc-input-0",
                    "y": gen_name(6)},
            outputs=[gen_name(8)],
            beta=1,
            alpha=1)
        if_layer_a.inputs["input-0"] = "fc-input-0"
        self.subgraph.inputs.append("fc-input-0")
        subgraph_block0.add_layer(
            "prim.equal", inputs={'input': gen_name(8)}, outputs=[gen_name(4)])
        subgraph_block1 = PaddleGraph()
        subgraph_block1.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(5)],
            value="middle_numpy[{}]".format(string(gen_name(5))))
        subgraph_block1.add_layer(
            "fluid.layers.transpose",
            inputs={"x": gen_name(5)},
            outputs=[gen_name(6)],
            perm=[1, 0])
        subgraph_block1.add_layer(
            "fluid.layers.matmul",
            inputs={"x": "fc-input-0",
                    "y": gen_name(6)},
            outputs=[gen_name(9)])
        if_layer_a.inputs["input-1"] = "fc-input-0"
        subgraph_block1.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(10)], value=True)
        subgraph_block1.add_layer("prim.if", {'input': gen_name(10)},
                                  [gen_name(11)])
        if_layer_b = subgraph_block1.layers[-1]
        subgraph_block1_block0 = PaddleGraph()
        subgraph_block1_block0.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(12)],
            value="middle_numpy[{}]".format(string(gen_name(12))))
        subgraph_block1_block0.add_layer(
            "prim.add",
            inputs={"x": gen_name(9),
                    "y": gen_name(12)},
            outputs=[gen_name(13)],
            alpha=1)
        if_layer_b.inputs["input-0"] = gen_name(9)
        subgraph_block1_block0.add_layer(
            "prim.equal",
            inputs={'input': gen_name(13)},
            outputs=[gen_name(11)])
        subgraph_block1_block1 = PaddleGraph()
        subgraph_block1_block1.add_layer(
            "prim.equal", inputs={'input': gen_name(9)},
            outputs=[gen_name(11)])
        if_layer_b.inputs["input-1"] = gen_name(9)
        subgraph_block1.add_layer(
            "prim.equal", inputs={'input': gen_name(11)},
            outputs=[gen_name(4)])
        if_layer_b.add_block(subgraph_block1_block0)
        if_layer_b.add_block(subgraph_block1_block1)
        if_layer_a.add_block(subgraph_block0)
        if_layer_a.add_block(subgraph_block1)
        self.subgraph.build(
            inputs={"input-0": "fc-input-0",
                    "input-1": "fc-input-0"})

    def replace_layer(self, graph, match_index):
        # match_index为[79, 80, 81, 82, 83, {84: [[0, 1, 2, 3, 4], [0, 1, 2, 3, {4: [[0, 1, 2], [0]]}, 5]] }]
        input_name = graph.layers[list(match_index[5].keys())[0]].blocks[
            0].layers[list(match_index[5].values())[0][0][3]].inputs["x"]
        output_name = graph.layers[list(match_index[5].keys())[0]].outputs[0]
        weight_name = graph.layers[list(match_index[5].keys())[0]].blocks[
            0].layers[list(match_index[5].values())[0][0][0]].attrs["value"][14:
                                                                             -2]
        bias_name = graph.layers[list(match_index[5].keys())[0]].blocks[
            0].layers[list(match_index[5].values())[0][0][2]].attrs["value"][14:
                                                                             -2]
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
        for i in range(len(self.subgraph.layers)):
            graph.layers.pop(match_index[0])
        new_layer = PaddleLayer(
            "fluid.dygraph.Linear",
            inputs={"input": input_name},
            outputs=[linear_name, output_name],
            **attrs)
        graph.layers.insert(match_index[0], new_layer)
        graph.clear_edges(graph)
        graph.build(show=True)
        return graph
