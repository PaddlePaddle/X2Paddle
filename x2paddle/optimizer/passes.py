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

from x2paddle.core.paddle_graph import PaddleGraph


class Pass(object):
    def __init__(self):
        self.pattern = PaddleGraph()
        self.build_pattern()


class Matcher(object):
    def __init__(self):
        self.unique_id_layer = dict()


class PyTorchMatcher(Matcher):
    def __init__(self):
        super(PyTorchMatcher, self).__init__()

    def match_pattern(self, pattern, graph, start_id):
        pattern_index = 0
        pattern_global_layers = pattern.get_global_layers()
        subgraph_global_layers = dict()
        graph_layers = dict(list(graph.layers.items())[start_id:])
        for layer_id, layer in graph_layers.items():
            pattern_layer = pattern.layers[list(pattern.layers.keys())[
                pattern_index]]
            if layer.kernel == pattern_layer.kernel:
                subgraph_global_layers[layer_id] = layer
                pattern_layer_id = pattern_layer.id
                if layer.kernel == "prim.constant":
                    if layer.attrs["value"] != pattern_layer.attrs["value"]:
                        return False
                elif layer.kernel == "fluid.layers.addmm":
                    if layer.attrs["beta"] != pattern_layer.attrs["beta"]:
                        return False
                    if layer.attrs["alpha"] != pattern_layer.attrs["alpha"]:
                        return False

                if layer_id in graph.edges_in:
                    if pattern_layer_id not in pattern.edges_in:
                        return False
                    else:
                        if len(graph.edges_in[layer_id]) != len(
                                pattern.edges_in[pattern_layer_id]):
                            return False
                    layer_in = graph.edges_in[layer_id]
                    pattern_layer_in = pattern.edges_in[pattern_layer_id]
                    for i in range(len(layer_in)):
                        layer_id_in = layer_in[i]
                        pattern_layer_id_in = pattern_layer_in[i]
                        if pattern_layer_id_in != -1:
                            pattern_global_layers_id = list(
                                pattern_global_layers.keys())
                            subgraph_global_layers_id = list(
                                subgraph_global_layers.keys())
                            if pattern_global_layers_id.index(pattern_layer_id_in) == \
                            subgraph_global_layers_id.index(layer_id_in):
                                # 判断pattern输入在pattern_global_layers_id的索引
                                # 和graph输入在subgraph_global_layers_id的索引一致
                                continue
                            return False

                if layer_id in graph.edges_out:
                    if pattern_layer_id not in pattern.edges_out:
                        if not set(pattern_layer.outputs).issubset(
                                pattern.outputs):
                            # 若pattern当前layer的输出是pattern的输出，则是正确的
                            return False
                    else:
                        if len(graph.edges_out[layer_id]) != len(
                                pattern.edges_out[pattern_layer_id]):
                            # 如果在每个节点edges_in相同的情况下，edges_out数目相同则说明无节点在subgraph外被用到
                            if not set(pattern_layer.outputs).issubset(
                                    pattern.outputs):
                                # 若pattern当前layer的输出是pattern的输出，则是正确的
                                return False

                if layer.kernel == "prim.if":
                    res = self.match_pattern(pattern_layer.blocks[0],
                                             layer.blocks[0], 0)
                    if res:
                        subgraph_global_layers.update(res)
                    else:
                        return False
                    res = self.match_pattern(pattern_layer.blocks[1],
                                             layer.blocks[1], 0)
                    if res:
                        subgraph_global_layers.update(res)
                    else:
                        return False
                pattern_index += 1
                if pattern_index == len(pattern.layers):
                    return subgraph_global_layers
            else:
                return False
