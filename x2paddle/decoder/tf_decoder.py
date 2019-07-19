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

from x2paddle.core.graph import GraphNode, Graph
from x2paddle.core.fluid_code import FluidCode
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.core.framework import attr_value_pb2
import tensorflow as tf
import copy


class TFGraphNode(GraphNode):
    def __init__(self, layer, layer_name=None):
        if layer_name is None:
            super(TFGraphNode, self).__init__(layer,
                                              layer.name.replace('/', '_'))
        else:
            super(TFGraphNode, self).__init__(layer,
                                              layer_name.replace('/', '_'))

        self.layer_type = layer.op
        self.fluid_code = FluidCode()

        self.dtype_map = {1: "float32", 3: "int32", 9: "int64"}

    @property
    def out_shapes(self):
        values = self.layer.attr["_output_shapes"].list.shape
        out_shapes = list()
        for value in values:
            shape = [dim.size for dim in value.dim]
            out_shapes.append(shape)
        return out_shapes

    @property
    def dtype(self):
        dtype = self.layer.attr["dtype"].type
        if dtype not in self.dtype_map:
            raise Exception("Dtype[{}] not in dtype_map".format(dtype))
        return self.dtype_map[dtype]

    @property
    def value(self):
        assert self.layer_type == "Const", "Only Const node has value."

        attr = self.layer.attr['value']
        field = getattr(attr, attr.WhichOneof('value'))
        return tensor_util.MakeNdarray(field)

    def get_attr(self, name):
        if name not in self.layer.attr:
            return None
        attr = self.layer.attr[name]
        field = attr.WhichOneof('value')
        value = getattr(attr, field) if field else None

        if isinstance(value, attr_value_pb2.AttrValue.ListValue):
            result = list(value.ListFields()[0][1])
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
            return value


class TFGraph(Graph):
    def __init__(self, model):
        super(TFGraph, self).__init__(model)
        self.identity_map = dict()

    def build(self):
        for layer in self.model.node:
            self.node_map[layer.name.replace('/', '_')] = TFGraphNode(layer)

        for layer_name, node in self.node_map.items():
            for in_node in node.layer.input:
                in_node = in_node.replace('/', '_')
                if in_node not in self.node_map:
                    if in_node.strip().split(':')[0] in self.node_map:
                        self.connect(in_node.strip().split(':')[0], layer_name)
                    else:
                        raise Exception(
                            'input[{}] of node[{}] does not exist in node_map'.
                            format(in_node, layer_name))
                else:
                    self.connect(in_node, layer_name)

        super(TFGraph, self).build()

        # tensorflow graph optimize
        self._remove_isolated_node()
        self._remove_identity_node()

    def get_node(self, node_name, copy=False):
        items = node_name.strip().split(':')
        items[0] = items[0].replace('/', '_')
        if items[0] in self.identity_map:
            items[0] = self.identity_map[items[0]]
        new_node_name = ":".join(items)
        return super(TFGraph, self).get_node(new_node_name, copy)

    def _remove_isolated_node(self):
        # delete isolated nodes
        isolated_nodes = list()
        for node_name in self.node_map.keys():
            if len(self.get_node(node_name).inputs) == 0 or len(
                    self.get_node(node_name).outputs) == 0:
                isolated_nodes.append(node_name)

        self.remove_node(node_name)

    def _remove_identity_node(self):
        identity_node = list()
        for node_name, node in self.node_map.items():
            if node.layer_type == "Identity":
                identity_node.append(node_name)

        for node_name in identity_node:
            node = self.get_node(node_name)
            # Remind: Only 1 input for Identity node
            input_node = self.get_node(node.inputs[0])

            # remove identity node from graph
            self.identity_map[node_name] = input_node.layer_name
            idx = input_node.outputs.index(node_name)
            del input_node.outputs[idx]

            output_names = node.outputs
            for output_name in output_names:
                output_node = self.get_node(output_name)
                idx = output_node.inputs.index(node_name)
                output_node.inputs[idx] = input_node.layer_name

            idx = self.topo_sort.index(node_name)
            del self.topo_sort[idx]


class TFDecoder(object):
    def __init__(self, pb_model):
        sess = tf.Session()
        with gfile.FastGFile(pb_model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        sess.run(tf.global_variables_initializer())

        self.tf_graph = TFGraph(sess.graph._as_graph_def(add_shapes=True)[0])
        self.tf_graph.build()
