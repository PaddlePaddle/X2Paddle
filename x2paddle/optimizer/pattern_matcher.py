# -*- coding:UTF-8 -*-
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
        elif match_kind == "op":
            self.detect_patterns_by_op(graph)
        self.remove_overlapped_match()
        return self.matches

    def detect_patterns_by_topo(self, graph):
        """ 找到与模式匹配的子图，
            并将子图的id以拓扑排序存放到subgraph_id2layers。
        """

        def get_subgraph(pattern, graph, start_index, is_subblock=False):
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
                            if pattern_index == 0 or is_subblock:
                                return False
                            else:
                                subgraph_id2layers.pop(layer_id)
                                continue
                        else:
                            if len(graph.edges_in[layer_id]) != len(
                                    pattern.edges_in[pattern_layer_id]):
                                if pattern_index == 0 or is_subblock:
                                    return False
                                else:
                                    subgraph_id2layers.pop(layer_id)
                                    continue
                        layer_in = graph.edges_in[layer_id]
                        pattern_layer_in = pattern.edges_in[pattern_layer_id]
                        for i in range(len(layer_in)):
                            layer_id_in = layer_in[i]
                            pattern_layer_id_in = pattern_layer_in[i]
                            if pattern_layer_id_in != -1:
                                subgraph_ids = list(subgraph_id2layers.keys())
                                if layer_id_in not in subgraph_ids:
                                    return False
                                if pattern_ids.index(pattern_layer_id_in) == \
                                subgraph_ids.index(layer_id_in):
                                    # 判断pattern输入在pattern_ids的索引
                                    # 和graph输入在subgraph_ids的索引一致
                                    continue
                                if pattern_index == 0 or is_subblock:
                                    return False
                                else:
                                    subgraph_id2layers.pop(layer_id)
                                    continue
                    # 判断subgraph中的节点是否被外部图使用到(如若被使用到则无效)
                    if layer_id in graph.edges_out:
                        if pattern_layer_id not in pattern.edges_out:
                            if "paddle.nn" in layer.kernel and "functional" not in layer.kernel:
                                pattern_layer_opt = pattern_layer.outputs[1:]
                            else:
                                pattern_layer_opt = pattern_layer.outputs
                            if not set(pattern_layer_opt).issubset(
                                    pattern.outputs):
                                # 若pattern当前layer的输出是pattern的输出，则是正确的
                                if pattern_index == 0 or is_subblock:
                                    return False
                                else:
                                    subgraph_id2layers.pop(layer_id)
                                    continue
                        else:
                            if len(graph.edges_out[layer_id]) != len(
                                    pattern.edges_out[pattern_layer_id]):
                                # 如果在每个节点edges_in相同的情况下，edges_out数目相同则说明无节点在subgraph外被用到
                                if "paddle.nn" in layer.kernel and "functional" not in layer.kernel:
                                    pattern_layer_opt = pattern_layer.outputs[
                                        1:]
                                else:
                                    pattern_layer_opt = pattern_layer.outputs
                                if not set(pattern_layer_opt).issubset(
                                        pattern.outputs):
                                    # 若pattern当前layer的输出是pattern的输出，则是正确的
                                    if pattern_index == 0 or is_subblock:
                                        return False
                                    else:
                                        subgraph_id2layers.pop(layer_id)
                                        continue
                            else:
                                layer_out = graph.edges_out[layer_id]
                                pattern_layer_out = pattern.edges_out[
                                    pattern_layer_id]
                                is_pop = False
                                for i in range(len(layer_out)):
                                    layer_id_out = layer_out[i]
                                    pattern_layer_id_out = pattern_layer_out[i]
                                    if layer_id_out != -1:
                                        if graph_layers[
                                                layer_id_out].kernel != pattern.layers[
                                                    pattern_layer_id_out].kernel:
                                            is_pop = True
                                            break
                                if is_pop:
                                    subgraph_id2layers.pop(layer_id)
                                    continue
                    # 当为控制流时的处理
                    if layer.kernel == "prim.if" or layer.kernel == "prim.loop":
                        if len(pattern_layer.blocks) != len(layer.blocks):
                            if pattern_index == 0 or is_subblock:
                                return False
                            else:
                                subgraph_id2layers.pop(layer_id)
                                continue
                        is_subblock_match = True
                        for i, b in enumerate(pattern_layer.blocks):
                            match_info = get_subgraph(
                                pattern_layer.blocks[i],
                                layer.blocks[i],
                                0,
                                is_subblock=True)
                            if match_info is not False:
                                subgraph_id2layers.update(match_info)
                            else:
                                is_subblock_match = False
                                break
                        if not is_subblock_match:
                            if pattern_index == 0 or is_subblock:
                                return False
                            else:
                                index = list(subgraph_id2layers.keys()).index(
                                    layer_id)
                                for key in list(subgraph_id2layers.keys())[
                                        index:]:
                                    subgraph_id2layers.pop(key)
                                continue
                    pattern_index += 1
                    if pattern_index == len(pattern.layers):
                        return subgraph_id2layers
                else:
                    if pattern_index == 0 or is_subblock:
                        return False
                    else:
                        continue
            if pattern_index == len(pattern.layers):
                return subgraph_id2layers
            return False

        for i, (layer_id, layer) in enumerate(graph.layers.items()):
            match_info = get_subgraph(self.pattern, graph, i)
            if match_info and match_info not in self.matches:
                self.matches.append(match_info)
            for j, block in enumerate(layer.blocks):
                if len(block.layers) > 0:
                    self.detect_patterns_by_topo(layer.blocks[j])

    def detect_patterns_by_edge(self, graph):
        """当遇见顺序没有强制规定的pattern时使用该方式
        """

        def get_subgraph(pattern, graph, start_index):
            pattern_id2layers = pattern.get_global_layers()
            pattern_ids = list(pattern_id2layers.keys())
            pattern_layer_id = pattern_ids[0]
            subgraph_id2layers = dict()
            layer_id = list(graph.layers.keys())[start_index]
            graph_layers = graph.layers

            def update(layer_id, pattern_layer_id):
                layer = graph_layers[layer_id]
                pattern_layer = pattern_id2layers[pattern_layer_id]
                if layer.kernel != pattern_layer.kernel:
                    return False
                subgraph_id2layers[layer_id] = layer

                if pattern.edges_in.get(pattern_layer_id, 0) != 0:
                    if len(pattern.edges_in[pattern_layer_id]) != \
                            len(graph.edges_in[layer_id]):
                        return False
                    for i, pattern_layer_id_in in enumerate(pattern.edges_in[
                            pattern_layer_id]):
                        if pattern_layer_id_in == -1:
                            continue
                        if pattern_layer_id_in in pattern_ids:
                            new_layer_id_in = graph.edges_in[layer_id][i]
                            if new_layer_id_in in subgraph_id2layers:
                                continue
                            update(new_layer_id_in, pattern_layer_id_in)
                if pattern.edges_out.get(pattern_layer_id, 0) != 0:
                    if layer_id not in graph.edges_out:
                        return False
                    if len(pattern.edges_out[pattern_layer_id]) != \
                            len(graph.edges_out[layer_id]):
                        return False
                    for i, pattern_layer_id_out in enumerate(pattern.edges_out[
                            pattern_layer_id]):
                        if pattern_layer_id_out in pattern_ids:
                            new_layer_id_out = graph.edges_out[layer_id][i]
                            if new_layer_id_out in subgraph_id2layers:
                                continue
                            update(new_layer_id_out, pattern_layer_id_out)

            while len(subgraph_id2layers) != len(pattern_id2layers):
                out = update(layer_id, pattern_layer_id)
                if out == False:
                    return False
                else:
                    if len(subgraph_id2layers) == len(pattern_id2layers):
                        return subgraph_id2layers
                    else:
                        return False

        for i, (layer_id, layer) in enumerate(graph.layers.items()):
            match_info = get_subgraph(self.pattern, graph, i)
            if match_info:
                self.matches.append(match_info)
            for j, block in enumerate(layer.blocks):
                if len(block.layers) > 0:
                    self.detect_patterns_by_edge(layer.blocks[j])

    def detect_patterns_by_op(self, graph):
        """ 当只匹配op时使用此方式。
        """

        def get_subgraph(pattern, graph, start_index):
            pattern_id2layers = pattern.get_global_layers()
            pattern_ids = list(pattern_id2layers.keys())
            pattern_layer_id = pattern_ids[0]
            subgraph_id2layers = dict()
            layer_id = list(graph.layers.keys())[start_index]
            graph_layers = graph.layers

            def update(layer_id, pattern_layer_id):
                layer = graph_layers[layer_id]
                pattern_layer = pattern_id2layers[pattern_layer_id]
                if layer.kernel != pattern_layer.kernel:
                    return False
                subgraph_id2layers[layer_id] = layer

            while len(subgraph_id2layers) != len(pattern_id2layers):
                out = update(layer_id, pattern_layer_id)
                if out == False:
                    return False
                else:
                    if len(subgraph_id2layers) == len(pattern_id2layers):
                        return subgraph_id2layers
                    else:
                        return False

        for i, (layer_id, layer) in enumerate(graph.layers.items()):
            match_info = get_subgraph(self.pattern, graph, i)
            if match_info:
                self.matches.append(match_info)
            for j, block in enumerate(layer.blocks):
                if len(block.layers) > 0:
                    self.detect_patterns_by_op(layer.blocks[j])

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
    def __init__(self):
        self.pattern = PaddleGraph()
        self.patterns = list()

    def operate(self, graph, match_kind="topo"):
        parameters = graph.parameters
        self.build_pattern()
        self.perform_pattern_matcher(graph, match_kind)
        for match in self.matches:
            first_layer_id = list(match.keys())[0]
            subgraph = get_subgraph("", first_layer_id, graph)
            self.insert_new_layer(subgraph, parameters, match)
        self.delete_match(graph)
        graph.build()

    def perform_pattern_matcher(self, graph, match_kind="topo"):
        """ 执行模式匹配，找到匹配的子图。
        """
        if len(self.patterns) > 0:
            self.matches = list()
            for pattern in self.patterns:
                pattern_matcher = PatternMatcher(pattern)
                self.matches.extend(pattern_matcher.operate(graph, match_kind))
        else:
            pattern_matcher = PatternMatcher(self.pattern)
            self.matches = pattern_matcher.operate(graph, match_kind)

    def delete_match(self, graph):
        """ 删除不需要的中间layer及其对应参数。
        """
        for match in self.matches:
            if len(match) == 0:
                continue
            first_layer_id = list(match.keys())[0]
            subgraph = get_subgraph("", first_layer_id, graph)
            for layer_id, layer in match.items():
                if layer_id in subgraph.layers:
                    # layer_id可能是属于子图的，此时删除父layer，即删除整个子图
                    subgraph.layers.pop(layer_id)
