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

import tensorflow as tf
from tensorflow_graph import TensorflowGraph
from tensorflow.python.framework import tensor_util
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
import numpy


class TensorflowCkptParser(object):
    def __init__(self,
                 meta_file,
                 checkpoint_file,
                 dest_nodes,
                 input_shape=None,
                 in_nodes=None,
                 input_format="NCHW".encode()):
        graph_def = None
        self.weights = None
        self.inputs = in_nodes
        self.outputs = dest_nodes
        sess = tf.Session()
        if meta_file is None:
            raise Exception("meta_file must be provided")
        new_saver = tf.train.import_meta_graph(meta_file)
        if checkpoint_file is not None:
            self.weights = dict()
            new_saver.restore(sess,
                              tf.train.latest_checkpoint(checkpoint_file))
            for var in tf.global_variables():
                value = var.eval(sess)
                self.weights[var.name.split(':')[0]] = value

        self.infer = ModelInfer(sess)
        graph_def, ver = tf.get_default_graph()._as_graph_def(add_shapes=True)

        if in_nodes is not None and input_shape is not None:
            graph_def = strip_unused_lib.strip_unused(
                input_graph_def=graph_def,
                input_node_names=in_nodes,
                output_node_names=dest_nodes,
                placeholder_type_enum=dtypes.float32.as_datatype_enum)

            for node in graph_def.node:
                if node.name in in_nodes:
                    index = in_nodes.index(node.name)
                    shape = [tf.Dimension(x) for x in input_shape[index]]
                    shape_proto = tf.TensorShape(shape).as_proto()
                    node.attr['_output_shapes'].list.shape.pop()
                    node.attr['_output_shapes'].list.shape.extend(
                        [shape_proto])
                    self.infer.gen_sample_data(node.name, input_shape[index])

            self.tf_graph = TensorflowGraph(graph_def)
        else:
            raise Exception('in_nodes and output_nodes need be provided')

        self.tf_graph.build(input_format)


class TensorflowPbParser(object):
    def __init__(self,
                 pb_file,
                 dest_nodes,
                 input_shape=None,
                 in_nodes=None,
                 input_format="NCHW".encode()):
        with open(pb_file, 'rb') as f:
            serialized = f.read()
        tf.reset_default_graph()
        original_graph_def = tf.GraphDef()
        original_graph_def.ParseFromString(serialized)
        self.inputs = list()
        self.outputs = dest_nodes

        sess = tf.Session(graph=tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        self.infer = ModelInfer(sess)

        original_graph_def = strip_unused_lib.strip_unused(
            input_graph_def=original_graph_def,
            input_node_names=in_nodes,
            output_node_names=dest_nodes,
            placeholder_type_enum=dtypes.float32.as_datatype_enum)

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(original_graph_def.SerializeToString())
        in_type_list = dict()
        for node in graph_def.node:
            if node.name in in_nodes:
                in_type_list[node.name] = node.attr['dtype'].type

        input_shape = list(input_shape)
        if not isinstance(input_shape[0], list):
            input_shape = [input_shape]

        input_map = dict()
        for i in range(len(input_shape)):
            if in_type_list[in_nodes[i]] == 1 or in_type_list[
                    in_nodes[i]] == 0:
                dtype = tf.float32
                x = tf.placeholder(dtype, shape=input_shape[i])
            elif in_type_list[in_nodes[i]] == 3:
                dtype = tf.int32
                x = tf.placehoder(dtype, shape=input_shape[i])
            else:
                raise Exception("Unexpected dtype for input, only support " \
                    "float32 and int32 now")
            input_map[in_nodes[i] + ":0"] = x
            self.inputs.append(x.name.split(':')[0])
            self.infer.gen_sample_data(x.name, input_shape[i])

        tf.import_graph_def(graph_def, name="", input_map=input_map)
        graph_def = tf.get_default_graph()._as_graph_def(add_shapes=True)[0]

        self.tf_graph = TensorflowGraph(graph_def)
        self.tf_graph.build(input_format)

        self.weights = dict()
        for node in graph_def.node:
            if node.op.lower() == "const":
                try:
                    node.attr['value'].tensor.tensor_content
                    weight = tensor_util.MakeNdarray(node.attr['value'].tensor)
                    self.weights[node.name] = weight
                except:
                    continue


class ModelInfer(object):
    """ Trick method for tensorflow2fluid
    There are some Operators in PaddlePaddle not support 
    tensor as parameter, like reshape/transpose, Because these
    parameters should be fixed in PaddlePaddle. So we 
    provide 'ModelInfer' here to solove this problem.
    """

    def __init__(self, sess):
        self.sess = sess
        self.inputs_sample_data = dict()

    def gen_sample_data(self, tensor_name, shape):
        self.inputs_sample_data[tensor_name] = list()
        if shape[0] is None or shape[0] < 0:
            for i in range(1, 4):
                data = numpy.random.random_sample([i] + shape[1:])
                self.inputs_sample_data[tensor_name].append(data)
        else:
            for i in range(1, 4):
                data = numpy.random.random_sample(shape)
                self.inputs_sample_data[tensor_name].append(data)

    def get_shape_tensor(self, layer, output_shape=None):
        """ return value of shape parameter
        return value of shape parameter which are tensor type 
        in tensorflow model
        """

        tensor_name = layer.name
        if len(tensor_name.split(':')) < 2:
            tensor_name = tensor_name + ':0'
        output_tensor = self.sess.graph.get_tensor_by_name(tensor_name)

        tensor_values = []
        for i in range(0, 3):
            inputs_tensors = dict()
            for name, values in self.inputs_sample_data.items():
                if len(name.split(':')) < 2:
                    name = name + ':0'
                tensor = self.sess.graph.get_tensor_by_name(name)
                inputs_tensors[tensor] = values[i]
            r, = self.sess.run([output_tensor], inputs_tensors)
            tensor_values.append(r.flatten())

        compare01 = (tensor_values[0] == tensor_values[1])
        compare12 = (tensor_values[1] == tensor_values[2])

        if compare01.all() and compare12.all():
            return tensor_values[0]

        if (compare01 == compare12).all():
            index = numpy.argwhere(compare01 == False).flatten()
            if index.shape[0] != 1:
                raise Exception("There's not only one unstable dimension")
            tensor_values[0][index[0]] = -1

            index = numpy.argwhere(tensor_values[0] < 0).flatten()
            if index.shape[0] > 2:
                raise Exception("There's more than two values less than zero")
            if index.shape[0] == 2:
                if output_shape is None:
                    raise Exception("Need output_shape parameter, " \
                        "get_shape_tensor(tensor_name, output_shape)")
                tensor_values[0][index[1]] = output_shape[index[1]]
            return tensor_values[0]
        else:
            raise Exception("Can not infer a stable shape tensor value")

    def get_tensor_shape(self, layer):
        shape = layer.attr['_output_shapes'].list.shape[0]
        shape = numpy.array([dim.size for dim in shape.dim])
        if numpy.argwhere(shape < 0).shape[0] <= 1:
            return shape
        tensor_name = layer.name
        if len(tensor_name.split(':')) < 2:
            tensor_name = tensor_name + ':0'
        output_tensor = self.sess.graph.get_tensor_by_name(tensor_name)

        shapes = []
        for i in range(0, 3):
            inputs_tensors = dict()
            for name, values in self.inputs_sample_data.items():
                if len(name.split(':')) < 2:
                    name = name + ':0'
                tensor = self.sess.graph.get_tensor_by_name(name)
                inputs_tensors[tensor] = values[i]
            r, = self.sess.run([output_tensor], inputs_tensors)
            shapes.append(numpy.array(r.shape))

        compare01 = (shapes[0] == shapes[1])
        compare12 = (shapes[1] == shapes[2])

        if compare01.all() and compare12.all():
            return shapes[0]

        if (compare01 == compare12).all():
            index = numpy.argwhere(compare01 == False).flatten()
            if index.shape[0] != 1:
                raise Exception("There's not only one unstable dimension")
            if index[0] != 0:
                raise Exception("Batch size not in the first dimension")
            shapes[0][0] = -1
            return shapes[0]
        else:
            raise Exception("Can not infer a stable tensor shape, failed!")

    def get_const_tensor_value(self, layer):
        tensor_name = layer.name
        if len(tensor_name.split(':')) < 2:
            tensor_name = tensor_name + ':0'
        output_tensor = self.sess.graph.get_tensor_by_name(tensor_name)

        result = []
        for i in range(0, 3):
            inputs_tensors = dict()
            for name, values in self.inputs_sample_data.items():
                if len(name.split(':')) < 2:
                    name = name + ':0'
                tensor = self.sess.graph.get_tensor_by_name(name)
                inputs_tensors[tensor] = values[i]
            r, = self.sess.run([output_tensor], inputs_tensors)
            result.append(r)

        compare01 = (result[0] == result[1])
        compare12 = (result[1] == result[2])

        if compare01.all() and compare12.all():
            return result[0]
        else:
            raise Exception("Can not infer a stable constant tensor value")
