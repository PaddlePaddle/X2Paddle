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

from utils import *
import collections


class GraphNode(object):
    def __init__(self, layer, layer_name=None):
        self.inputs = list()
        self.outputs = list()
        self.layer = layer
        self.ref_name = None
        self.output_name = None
        if layer_name is not None:
            self.layer_name = layer_name
        else:
            self.layer_name = layer.name

    def __hash__(self):
        return hash(self.layer.name)

    def __eq__(self, other):
        if self.layer.name == other.layer.name:
            return True
        return False


class Graph(object):
    def __init__(self, model):
        self.node_map = collections.OrderedDict()
        self.input_nodes = list()
        self.output_nodes = list()
        self.topological_sort = list()
        self.model = model
        self.name_generator = NameGenerator()

    def build(self, input_format):
        self._make_input_nodes()
        self._make_output_nodes()
        self._get_topological_sort()
        self._gen_newname_for_nodes()

    def _make_input_nodes(self):
        for name, node in self.node_map.items():
            if len(node.outputs) == 0 and len(node.inputs) == 0:
                continue
            node.left_inputs = len(node.inputs)
            if len(node.inputs) == 0:
                self.input_nodes.append(name)

    def _make_output_nodes(self):
        for name, node in self.node_map.items():
            if len(node.outputs) == 0 and len(node.inputs) == 0:
                continue
            if len(node.outputs) == 0:
                self.output_nodes.append(name)

    def _get_topological_sort(self):
        self.topological_sort = self.input_nodes[:]
        idx = 0
        while idx < len(self.topological_sort):
            current_node = self.node_map[self.topological_sort[idx]]
            for next_node in current_node.outputs:
                next_node_info = self.node_map[next_node.layer_name]
                next_node_info.left_inputs -= 1
                if next_node_info.left_inputs == 0:
                    self.topological_sort.append(next_node.layer_name)
            idx += 1

    def _gen_newname_for_nodes(self):
        for node_name in self.topological_sort:
            node = self.node_map[node_name]
            ref_name = self.name_generator.get_name(node)

            if node.layer_type == 'split' or node.layer_type == 'splitv':
                index = '0'
                if len(node_name.split(':')) == 2:
                    index = node_name.split(':')[-1]
                ref_name += '[{}]'.format(index)

            self.node_map[node.layer.name].ref_name = ref_name
            self.node_map[node.layer.name].output_name = ref_name.split('[')[0]

        for node_name, node in self.node_map.items():
            ref_name = self.name_generator.get_name(node)
            if node.layer_type == 'split' or node.layer_type == 'splitv':
                index = '0'
                if len(node_name.split(':')) == 2:
                    index = node_name.split(':')[-1]
                ref_name += '[{}]'.format(index)
                self.node_map[node_name].ref_name = ref_name
                self.node_map[node_name].output_name = ref_name.split('[')[0]

    def get_node(self, name):
        if name not in self.node_map:
            raise Exception("Graph doesn't have node [%s]." % name)
        else:
            return self.node_map[name]

    def _make_connection(self, src, dst):
        if src.layer_name == dst.layer_name or src.layer_name not in \
            self.node_map or dst.layer_name not in self.node_map:
            raise Exception('Warning: Node not exist or there is a self-loop')
        self.node_map[dst.layer_name].inputs.append(src)
        self.node_map[src.layer_name].outputs.append(dst)
