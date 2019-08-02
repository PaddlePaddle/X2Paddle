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
import copy as cp
import numpy
import sys


class TFGraphNode(GraphNode):
    def __init__(self, layer, layer_name=None):
        if layer_name is None:
            super(TFGraphNode,
                  self).__init__(layer,
                                 layer.name.replace('/', '_').replace('-', '_'))
        else:
            super(TFGraphNode,
                  self).__init__(layer,
                                 layer_name.replace('/', '_').replace('-', '_'))

        self.layer_type = layer.op
        self.fluid_code = FluidCode()

        self.dtype_map = {1: "float32", 3: "int32", 4: "int8", 9: "int64"}

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
        self.multi_out_ops = ['Split', 'SplitV']

    def build(self):
        for layer in self.model.node:
            self.node_map[layer.name.replace('/', '_').replace(
                '-', '_')] = TFGraphNode(layer)

        for layer_name, node in self.node_map.items():
            for in_node in node.layer.input:
                in_node = in_node.replace('/', '_').replace('-', '_')
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
        items[0] = items[0].replace('/', '_').replace('-', '_')
        if items[0] in self.identity_map:
            items[0] = self.identity_map[items[0]]
        new_node_name = ":".join(items)
        node = super(TFGraph, self).get_node(new_node_name, copy)
        if len(items) == 1 and node.layer_type in self.multi_out_ops:
            node.index = 0
        return node

    def _remove_isolated_node(self):
        # delete isolated nodes
        isolated_nodes = list()
        for node_name in self.node_map.keys():
            if len(self.get_node(node_name).inputs) == 0 and len(
                    self.get_node(node_name).outputs) == 0:
                isolated_nodes.append(node_name)

        for node_name in isolated_nodes:
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

            if node_name in self.output_nodes:
                idx = self.output_nodes.index(node_name)
                self.output_nodes[idx] = input_node.layer_name


class TFDecoder(object):
    def __init__(self, pb_model):
        self.sess = tf.Session()
        self.input_info = dict()
        with gfile.FastGFile(pb_model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_map = self._check_input_shape(graph_def)
            self._fix_output_shape(graph_def)
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='', input_map=input_map)


#        for node in graph_def.node:
#            print(node.name, node.op, node.input)

        self.sess.run(tf.global_variables_initializer())

        self.tf_graph = TFGraph(
            self.sess.graph._as_graph_def(add_shapes=True)[0])
        self.tf_graph.build()

    def _fix_output_shape(self, graph):
        for i in range(len(graph.node)):
            node = graph.node[i]
            if node.op == "swish_f32":
                graph.node[i].attr['_disable_call_shape_inference'].b = False

    def _check_input_shape(self, graph_def):
        numpy.random.seed(13)
        graph_def = cp.deepcopy(graph_def)
        input_map = dict()
        for layer in graph_def.node:
            if layer.op != "Placeholder":
                continue
            graph_node = TFGraphNode(layer)
            dtype = graph_node.dtype

            need_define_shape = 0
            if not graph_node.get_attr("shape"):
                need_define_shape = 1
            else:
                value = graph_node.layer.attr["shape"].shape
                shape = [dim.size for dim in value.dim]
                if shape.count(-1) > 1:
                    need_define_shape = 2

            if need_define_shape > 0:
                if need_define_shape == 1:
                    print(
                        "\nUnknown shape for input tensor[tensor name: \"{}\"]".
                        format(layer.name))
                else:
                    print(
                        "\nShape[now is {}] for input tensor[tensor name: \"{}\"] not support yet"
                        .format(shape, layer.name))
                print(
                    "Use your keyboard type the shape of input tensor below :)")

                right_shape_been_input = False
                while not right_shape_been_input:
                    shape = input("Shape of Input(e.g. None,224,224,3): ")
                    if shape.count("None") > 1:
                        print("Only 1 dimension can be None, type again:)")
                    else:
                        right_shape_been_input = True

                shape = [
                    None if dim == "None" else int(dim)
                    for dim in shape.strip().split(',')
                ]
                assert shape.count(None) <= 1, "Only one dimension can be None"
                x2paddle_input = tf.placeholder(dtype=dtype,
                                                shape=shape,
                                                name="x2paddle_{}".format(
                                                    layer.name))
                input_map["{}:0".format(layer.name)] = x2paddle_input
                shape[shape.index(None)] = -1
                #                self.input_example_data["x2paddle_{}".format(layer.name)] = numpy.random.random_sample(shape).astype(dtype)
                self.input_info["x2paddle_{}".format(layer.name)] = (shape,
                                                                     dtype)
            else:
                value = graph_node.layer.attr["shape"].shape
                shape = [dim.size for dim in value.dim]
                #                self.input_example_data[graph_node.layer_name] = numpy.random.random_sample(shape).astype(dtype)
                self.input_info[graph_node.layer_name] = (shape, dtype)

        return input_map

    # trick method
    # should be removed after PaddlePaddle V1.6 been released
    def infer_tensor(self, graph_node):
        print("========== Use infer_tensor for tensor: ", graph_node.layer.name)
        if hasattr(graph_node, "index"):
            tensor_name = graph_node.layer.name + ":{}".format(graph_node.index)
        else:
            tensor_name = graph_node.layer.name + ":0"
        feed = dict()
        for input_name, info in self.input_info.items():
            (shape, dtype) = cp.deepcopy(info)
            input_tensor = self.sess.graph.get_tensor_by_name(input_name + ":0")
            if shape.count(-1) > 0:
                shape[shape.index(-1)] = 2
            feed[input_tensor] = numpy.random.random_sample(shape)
        output_tensor = self.sess.graph.get_tensor_by_name(tensor_name)
        return self.sess.run([output_tensor], feed)[0]

    def infer_shape_tensor(self, graph_node, out_shape=None):
        print("========== Use infer_shape_tensor for tensor: ",
              graph_node.layer.name)
        if hasattr(graph_node, "index"):
            tensor_name = graph_node.layer.name + ":{}".format(graph_node.index)
        else:
            tensor_name = graph_node.layer.name + ":0"
        feed = dict()
        batch_size = [2, 3, 5]
        results = list()
        for b in batch_size:
            for input_name, info in self.input_info.items():
                (shape, dtype) = cp.deepcopy(info)
                input_tensor = self.sess.graph.get_tensor_by_name(input_name +
                                                                  ":0")
                if shape.count(-1) > 0:
                    shape[shape.index(-1)] = b
                feed[input_tensor] = numpy.random.random_sample(shape)
            output_tensor = self.sess.graph.get_tensor_by_name(tensor_name)
            results.append(self.sess.run([output_tensor], feed)[0].flatten())

        compare01 = (results[0] == results[1])
        compare12 = (results[1] == results[2])

        if compare01.all() and compare12.all():
            return results[0].tolist()

        if (compare01 == compare12).all():
            index = numpy.argwhere(compare01 == False).flatten()
            if index.shape[0] != 1:
                raise Exception("There's not only one unstable dimension")
            results[0][index[0]] = -1

            index = numpy.argwhere(results[0] < 0).flatten()
            if index.shape[0] > 2:
                print("Warning: More than two dimension less than zero")
            if index.shape[0] == 2 and out_shape is not None:
                if out_shape[index[1]] > 0:
                    results[0][index[1]] = out_shape[index[1]]
                else:
                    results[0][index[0]] = out_shape[index[0]]
            return results[0].tolist()
        else:
            raise Exception("Couldn't infer a stable shape shape tensor value")
