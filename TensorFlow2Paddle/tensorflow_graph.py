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
        self.dataformat = 'NCHW'

    @property
    def type(self):
        return self.layer.op.lower()

    @property
    def name(self):
        return self.layer.name

    # TODO
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
    def __init__(self, model):
        super(TensorflowGraph, self).__init__(model)
        self.model = model

    def build(self):
        for i, layer in enumerate(self.model.node):
            self.node_map[layer.name] = TensorflowGraphNode(layer)
            for pred in layer.input:
                if pred not in self.node_map:
                    raise Exception('input: {} not in node_map'.format(pred))

                self._make_connection(pred, layer.name)

        super(TensorflowGraph, self).build()
        self._check_dataformat()

    # check the dataformat of network
    def _check_dataformat(self):
        ss = list()
        for i in range(0, len(self.topological_sort)):
            current_node = self.node_map[self.topological_sort[i]]
            if current_node.type == 'conv2d':
                s = current_node.layer.attr['data_format'].s
                if s != 'NHWC' and s != 'NCHW':
                    raise Exception('Unkown dataformat {}'.format(s))
                ss.append(s)

        if len(set(ss)) > 1:
            raise Exception("Two type of dataformat exist in this model")

        if len(set(ss)) == 0:
            return

        for i in range(0, len(self.topological_sort)):
            current_node = self.node_map[self.topological_sort[i]]
            current_node.dataformat = ss[0]
