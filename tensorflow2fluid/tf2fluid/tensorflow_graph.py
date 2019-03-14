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
from utils import *


class TensorflowGraphNode(GraphNode):
    dtype_map = {1: "float32", 3: "int32", 9: "int64"}

    def __init__(self, layer, input_format, layer_name=None):
        super(TensorflowGraphNode, self).__init__(layer, layer_name)
        self.codes = list()
        self.code = FluidCode()
        self.ref_as_const = 0
        self.data_format = input_format

    @property
    def layer_type(self):
        return self.layer.op.lower()

    @property
    def shape_dim_size(self):
        shape = self.layer.attr['_output_shapes']
        return len(shape.list.shape[0].dim)

    @property
    def dtype(self):
        dtype = self.get_attr("dtype")
        if dtype in self.dtype_map:
            dtype = self.dtype_map[dtype]
        else:
            raise Exception("Unknow dtype: {}".format(dtype))
        return dtype

    def get_attr(self, name, default_value=None):
        if name in self.layer.attr:
            attr = self.layer.attr[name]
            field = attr.WhichOneof('value')
            val = getattr(attr, field) if field else default_value
            if isinstance(val, attr_value_pb2.AttrValue.ListValue):
                result = list(val.ListFields()[0][1])
                for i in range(len(result)):
                    if isinstance(result[i], int):
                        result[i] = int(result[i])
                    try:
                        if isinstance(result[i], long):
                            result[i] = int(result[i])
                    except:
                        pass
                return result
            else:
                return val if isinstance(val, bytes) else val
        else:
            return default_value

    def clear_code(self):
        self.code.clear()


class TensorflowGraph(Graph):
    useless_type = ['identity', 'placeholderwithdefault', 'switch', 'merge']

    def __init__(self, tf_graph):
        super(TensorflowGraph, self).__init__(tf_graph)
        self.tf_graph = tf_graph

    def build(self, input_format):
        skip_node = set(['const'])
        for i, layer in enumerate(self.tf_graph.node):
            self.node_map[layer.name] = TensorflowGraphNode(
                layer, input_format)

        for i, layer in enumerate(self.tf_graph.node):
            if layer.op.lower() in skip_node:
                continue
            for pred in layer.input:
                if pred not in self.node_map and pred.split(
                        ':')[0] in self.node_map:
                    pred_node = self.node_map[pred.split(':')[0]]
                    if pred_node.layer_type == "switch":
                        self._make_connection(pred_node,
                                              self.node_map[layer.name])
                    elif pred_node.layer_type == "split" or \
                        pred_node.layer_type == "splitv":
                        self.node_map[pred] = TensorflowGraphNode(
                            pred_node.layer, input_format, pred)
                        self._make_connection(self.node_map[pred],
                                              self.node_map[layer.name])
                        self._make_connection(pred_node, self.node_map[pred])
                    else:
                        raise Exception("Unsupported situation(name:[{}], \
                            OP[{}])".format(node.layer_name, node.layer_type))

                elif pred in self.node_map:
                    self._make_connection(self.node_map[pred],
                                          self.node_map[layer.name])

                else:
                    raise Exception("input: {} not in node_map".format(pred))
        super(TensorflowGraph, self).build(input_format)

        self._process_useless_nodes()
        self._check_dataformat(input_format)

    def _check_dataformat(self, input_format):
        for i in range(len(self.topological_sort)):
            current_node = self.node_map[self.topological_sort[i]]
            if 'data_format'.encode() in current_node.layer.attr:
                s = current_node.layer.attr['data_format'].s
                if s != NHWC and s != NCHW:
                    raise Exception('Unkown dataformat {}'.format(s))
                self.set_data_format(current_node, s)

    def _process_useless_nodes(self):
        remove_index = list()
        for i in range(len(self.topological_sort)):
            name = self.topological_sort[i]
            current_node = self.node_map[name]
            if current_node.layer_type in self.useless_type:
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
        for i in range(len(remove_index)):
            del self.topological_sort[remove_index[i]]

    def set_data_format(self, node, data_format):
        assert data_format == 'NHWC'.encode() or data_format == 'NCHW'.encode()
        if node.data_format == data_format:
            return
        node.data_format = data_format
        if len(node.outputs) == 0:
            return
        for output in node.outputs:
            self.set_data_format(output, data_format)
