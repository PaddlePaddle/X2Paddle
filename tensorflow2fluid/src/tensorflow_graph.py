#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from graph import GraphNode, Graph
from tensorflow.core.framework import attr_value_pb2


class TensorflowGraphNode(GraphNode):
    def __init__(self, layer):
        super(TensorflowGraphNode, self).__init__(layer)
        self.codes = list()
        self.data_format = 'NCHW'

    @property
    def layer_type(self):
        return self.layer.op.lower()

    @property
    def layer_name(self):
        return self.layer.name

    def get_attr(self, name, default_value=None):
        if name in self.layer.attr:
            attr = self.layer.attr[name]
            field = attr.WhichOneof('value')
            val = getattr(attr, field) if field else default_value
            if isinstance(val, attr_value_pb2.AttrValue.ListValue):
                return list(val.ListFields()[0][1])
            else:
                return val.decode('utf-8') if isinstance(val, bytes) else val
        else:
            return default_value


class TensorflowGraph(Graph):
    def __init__(self, tf_graph):
        super(TensorflowGraph, self).__init__(tf_graph)
        self.tf_graph = tf_graph

    def build(self):
        skip_node = set(['const'])
        for i, layer in enumerate(self.tf_graph.node):
            self.node_map[layer.name] = TensorflowGraphNode(layer)
        for i, layer in enumerate(self.tf_graph.node):
            if layer.op.lower() in skip_node:
                continue
            for pred in layer.input:
                if pred not in self.node_map and pred.split(
                        ':')[0] in self.node_map:
                    node = self.node_map[pred.split(':')[0]]
                    if node.layer_type == "switch":
                        self._make_connection(node, self.node_map[layer.name])
                    else:
                        raise Exception("Need to fix here")

                elif pred in self.node_map:
                    self._make_connection(self.node_map[pred],
                                          self.node_map[layer.name])

                else:
                    raise Exception("input: {} not in node_map".format(pred))
        super(TensorflowGraph, self).build()

        self._remove_useless_nodes()
        self._check_dataformat()

    def _check_dataformat(self):
        ss = list()
        for i in range(0, len(self.topological_sort)):
            current_node = self.node_map[self.topological_sort[i]]
            if 'data_format' in current_node.layer.attr:
                s = current_node.layer.attr['data_format'].s
                if s != 'NHWC' and s != 'NCHW':
                    raise Exception('Unkown dataformat {}'.format(s))
                ss.append(s)

        if len(set(ss)) > 1:
            raise Exception("Two type of dataformat exist in this model")

        if len(set(ss)) == 0:
            return

        for k, v in self.node_map.items():
            self.node_map[k].data_format = ss[0]

    def _remove_useless_nodes(self):
        useless_type = set(
            ['identity', 'placeholderwithdefault', 'switch', 'merge'])
        remove_index = list()
        for i in range(0, len(self.topological_sort)):
            name = self.topological_sort[i]
            current_node = self.node_map[name]
            if current_node.layer_type in useless_type:
                input = current_node.inputs[0]
                for node in current_node.outputs:
                    for k in range(0, len(node.inputs)):
                        if node.inputs[k] == current_node:
                            node.inputs[k] = input
                            if node not in input.outputs:
                                input.outputs.append(node)
                input.outputs.remove(current_node)
                del self.node_map[name]
                if name in self.output_nodes:
                    self.output_nodes.remove(name)
                if name in self.input_nodes:
                    self.input_nodes.remove(name)
                remove_index.append(i)

        remove_index.sort(reverse=True)
        for i in range(0, len(remove_index)):
            del self.topological_sort[remove_index[i]]
