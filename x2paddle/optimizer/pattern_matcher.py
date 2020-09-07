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

from x2paddle.core.program import PaddleGraph


class PatternMatcher(object):
    def __init__(self, pattern):
        self.pattern = pattern
        # matches的每个match是按照拓扑排序组成layer的dict
        self.matches = list()

    def operate(self, graph, match_kind="topo"):
        if match_kind == "topo":
            self.detect_patterns_by_topo(graph)
        elif match_kind == "edge":
            self.detect_patterns_by_edge(graph)
        self.remove_overlapped_match()
        return self.matches

    def detect_patterns_by_topo(self, graph):
        """ 找到与模式匹配的子图，
            并将子图的id以拓扑排序存放到subgraph_id2layers。
        """

        def get_subgraph(pattern, graph, start_index):
            pattern_index = 0
            pattern_id2layers = pattern.get_global_layers()
            pattern_ids = list(pattern_id2layers.keys())
            subgraph_id2layers = dict()
            graph_layers = dict(list(graph.layers.items())[start_index:])
            for layer_id, layer in graph_layers.items():
                pattern_layer = pattern.layers[list(pattern.layers.keys())[
                    pattern_index]]
                if layer.kernel == pattern_layer.kernel:
                    subgraph_id2layers[layer_id] = layer
                    pattern_layer_id = pattern_layer.id
                    # 判断输入连接是否一致
                    if layer_id in graph.edges_in:
                        if pattern_layer_id not in pattern.edges_in:
                            print("1--")
                            return False
                        else:
                            if len(graph.edges_in[layer_id]) != len(
                                    pattern.edges_in[pattern_layer_id]):
                                print("2--")
                                return False
                        layer_in = graph.edges_in[layer_id]
                        pattern_layer_in = pattern.edges_in[pattern_layer_id]
                        for i in range(len(layer_in)):
                            layer_id_in = layer_in[i]
                            pattern_layer_id_in = pattern_layer_in[i]
                            if pattern_layer_id_in != -1:
                                subgraph_ids = list(subgraph_id2layers.keys())
                                if pattern_ids.index(pattern_layer_id_in) == \
                                subgraph_ids.index(layer_id_in):
                                    # 判断pattern输入在pattern_ids的索引
                                    # 和graph输入在subgraph_ids的索引一致
                                    continue
                                print("3--")
                                return False
                    # 判断subgraph中的节点是否被外部图使用到(如若被使用到则无效)
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
                    # 当为控制流时的处理
                    if layer.kernel == "prim.if" or layer.kernel == "prim.loop":
                        if len(pattern_layer.blocks) != len(layer.blocks):
                            return False
                        for i, b in enumerate(pattern_layer.blocks):
                            match_info = get_subgraph(pattern_layer.blocks[i],
                                                      layer.blocks[i], 0)
                            if match_info is not False:
                                subgraph_id2layers.update(match_info)
                            else:
                                return False
                    pattern_index += 1
                    if pattern_index == len(pattern.layers):
                        return subgraph_id2layers
                else:
                    return False
            return subgraph_id2layers

        for i, (layer_id, layer) in enumerate(graph.layers.items()):
            match_info = get_subgraph(self.pattern, graph, i)
            if match_info:
                self.matches.append(match_info)
            for j, block in enumerate(layer.blocks):
                if len(block.layers) > 0:
                    self.detect_patterns_by_topo(layer.blocks[j])

    def detect_patterns_by_edge(self, graph):
        """当遇见顺序没有强制规定的pattern时使用该方式
        """
        pass

    def remove_overlapped_match(self):
        """ 如果2个子图有重叠，只取前一个子图。
        """
        match_ids = []
        for i, match in enumerate(self.matches):
            is_overlapped = False
            for id in match.keys():
                if id in match_ids:
                    self.matches.pop(i)
                    is_overlapped = True
                    break
            if not is_overlapped:
                match_ids.extend(list(match.keys()))


def get_subgraph(prefix_layer_id, suffix_layer_id, graph):
    """ 根据prefix_layer_id和suffix_layer_id获取需要子图。
        Args:
            prefix_layer_id (str): 起初为一个空字符串，之后为suffix_layer_id分割出来的前缀。
            suffix_layer_id (str): 起初为以一个layer的id，之后将分割部分给prefix_layer_id；例如”57.0.1“；
            graph (x2paddle.core.program.PaddleGraph): 需要进行pass的子图。
    """
    id_part = suffix_layer_id.split(".")
    if len(id_part) == 1:
        return graph
    if prefix_layer_id == "":
        layer_id = id_part[0]
        prefix_layer_id += ".".join(id_part[:2])
    else:
        layer_id = prefix_layer_id + "." + id_part[0]
        prefix_layer_id += ("." + ".".join(id_part[:2]))
    subgraph = graph.layers[layer_id].blocks[int(id_part[1])]
    suffix_layer_id = ".".join(id_part[2:])
    return get_subgraph(prefix_layer_id, suffix_layer_id, subgraph)


class FuseBase(object):
    def __init__(self, graph_type):
        self.pattern = PaddleGraph(graph_type=graph_type)

    def operate(self, graph, match_kind="topo"):
        parameters = graph.parameters
        self.build_pattern()
        self.perform_pattern_matcher(graph, match_kind)
        for match in self.matches:
            first_layer_id = list(match.keys())[0]
            subgraph = get_subgraph("", first_layer_id, graph)
            self.insert_new_layer(subgraph, parameters, match)
        self.delete_inter_layer(graph)
        graph.build()

    def perform_pattern_matcher(self, graph, match_kind="topo"):
        """ 执行模式匹配，找到匹配的子图。
        """
        pattern_matcher = PatternMatcher(self.pattern)
        self.matches = pattern_matcher.operate(graph, match_kind)

    def delete_inter_layer(self, graph):
        """ 删除不需要的中间layer及其对应参数。
        """
        for match in self.matches:
            first_layer_id = list(match.keys())[0]
            subgraph = get_subgraph("", first_layer_id, graph)
            for layer_id, layer in match.items():
                if layer.kernel == "fluid.dygraph.base.to_variable" and \
                layer.attrs["value"].startswith("params["):
                    param_name = layer.attrs["value"][8:-2]
                    if param_name in graph.parameters:
                        graph.parameters.pop(param_name)
                if layer_id in subgraph.layers:
                    # layer_id可能是属于子图的，此时删除父layer，即删除整个子图
                    subgraph.layers.pop(layer_id)
