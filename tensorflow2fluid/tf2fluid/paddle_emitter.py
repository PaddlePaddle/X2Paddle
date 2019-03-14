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

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from utils import *
from functools import *
from six import string_types as _string_types
import framework_pb2 as framework
import logging
import math
import struct
import numpy
logging.basicConfig(level=logging.DEBUG)


class PaddleEmitter(object):
    def __init__(self, parser, save_dir):
        self.graph = parser.tf_graph
        self.weights = parser.weights
        self.infer = parser.infer
        self.inputs_sample_data = dict()
        self.save_dir = save_dir
        self.body_code = ""
        self.tab = " " * 4

        self.outputs = parser.outputs
        self.inputs = parser.inputs
        outputs = list()
        for output in self.outputs:
            while True:
                if output in self.graph.identity_relation:
                    output = self.graph.identity_relation[output]
                else:
                    break
            outputs.append(output)
        self.outputs = outputs

    @staticmethod
    def compute_padding_size(in_size, filter_size, stride):
        new_size = int(math.ceil(in_size * 1.0 / stride))
        pad_size = (new_size - 1) * stride + filter_size - in_size
        pad_0 = int(pad_size / 2)
        pad_1 = pad_size - pad_0
        return [pad_0, pad_1]

    def check_op(self, node_name_list):
        uncovered_ops = set()
        for name in node_name_list:
            node = self.graph.get_node(name)
            if len(node.inputs) == 0 and len(node.outputs) == 0:
                continue
            if not hasattr(self, "emit_" + node.layer_type):
                uncovered_ops.add(node.layer_type)
        if len(uncovered_ops) > 0:
            logging.error("{} OP are not supported".format(len(uncovered_ops)))
            for op in uncovered_ops:
                logging.error("Unsupported OP: {}".format(op))
            return False
        return True

    # trick method to solve NHWC problem
    def get_axis(self, node1, node2):
        shape1 = node1.shape_dim_size
        shape2 = node2.shape_dim_size
        if shape1 == 4 and shape2 == 1 and node1.data_format == NHWC:
            axis = 1
        elif shape2 == 4 and shape1 == 1 and node2.data_format == NHWC:
            axis = 1
        else:
            axis = -1
        return axis

    def elementwise(self, node, op):
        data1 = node.inputs[0]
        data2 = node.inputs[1]
        axis = self.get_axis(data1, data2)
        shape1 = self.infer.get_tensor_shape(data1.layer)
        shape2 = self.infer.get_tensor_shape(data2.layer)

        op = "elementwise_" + op
        if shape2.shape[0] == shape1.shape[0]:
            if (shape1 == shape2).all():
                param_attr = {
                    'x': data1.ref_name,
                    'y': data2.ref_name,
                }
                node.code.add_layer(op, None, node.output_name, param_attr)
                return

            index1_not_one = list(numpy.argwhere(shape1 != 1).flatten())
            index1_one = list(numpy.argwhere(shape1 == 1).flatten())
            perm1 = range(shape1.shape[0])
            perm2 = range(shape1.shape[0])
            if len(index1_one) != 0:
                perm1 = index1_not_one + index1_one

            index2_not_one = list(numpy.argwhere(shape2 != 1).flatten())
            index2_one = list(numpy.argwhere(shape2 == 1).flatten())
            if len(index2_one) != 0:
                perm2 = index2_not_one + index2_one

            perm = list(numpy.array(perm1)[numpy.array(perm2)])
            if perm != range(shape1.shape[0]):
                param_attr = {"perm": perm}
                node.code.add_layer("transpose", data1.ref_name, "temp1",
                                    param_attr)
                node.code.add_layer("transpose", data2.ref_name, "temp2",
                                    param_attr)
                if len(index2_one) > len(index1_one):
                    param_attr = {"x": "temp1", "y": "temp2"}
                else:
                    param_attr = {"x": "temp2", "y": "temp1"}
                node.code.add_layer(op, None, node.output_name, param_attr)
                perm = sorted(range(len(perm)), key=lambda k: perm[k])
                param_attr = {"perm": perm}
                node.code.add_layer("transpose", node.output_name,
                                    node.output_name, param_attr)
            else:
                if len(index2_one) > len(index1_one):
                    param_attr = {"x": data1.ref_name, "y": data2.ref_name}
                else:
                    param_attr = {"x": data2.ref_name, "y": data1.ref_name}
                node.code.add_layer(op, None, node.output_name, param_attr)
        else:
            param_attr = {
                "x": data1.ref_name,
                "y": data2.ref_name,
                "axis": axis
            }
            if shape2.shape[0] > shape1.shape[0]:
                param_attr = {
                    "x": data2.ref_name,
                    "y": data1.ref_name,
                    "axis": axis
                }
            node.code.add_layer(op, None, node.output_name, param_attr)

    def export_weights(self, weight, paddle_var_name, dir):
        self.save_var_set.add(paddle_var_name)
        numpy_dtype_map = {
            "int16": framework.VarType.INT16,
            "int32": framework.VarType.INT32,
            "int64": framework.VarType.INT64,
            "float16": framework.VarType.FP16,
            "float32": framework.VarType.FP32,
            "float64": framework.VarType.FP64
        }
        struct_write_format = {
            "int16": "h",
            "int32": "i",
            "int64": "q",
            "float16": "e",
            "float32": "f",
            "float64": "d"
        }
        shape = weight.shape
        filew = open(dir + "/" + paddle_var_name, "wb")
        filew.write(struct.pack('i', 0))
        filew.write(struct.pack('L', 0))
        filew.write(struct.pack('i', 0))
        tensor_desc = framework.VarType.TensorDesc()
        if str(weight.dtype) in numpy_dtype_map:
            tensor_desc.data_type = numpy_dtype_map[str(weight.dtype)]
        else:
            raise Exception("Unexpected array dtype [{}]".format(weight.dtype))
        tensor_desc.dims.extend(shape)
        desc_size = tensor_desc.ByteSize()
        filew.write(struct.pack('i', desc_size))
        filew.write(tensor_desc.SerializeToString())
        tensor_size = reduce(lambda x, y: x * y, shape)
        weight = weight.flatten()
        for i in range(0, tensor_size):
            filew.write(
                struct.pack(struct_write_format[str(weight.dtype)], weight[i]))
        filew.close()

    @property
    def header_code(self):
        code = list()
        code.append("import paddle.fluid.layers as layers")
        code.append("import paddle.fluid as fluid")
        code.append("import numpy")
        code.append("")
        code.append("class Model(object):")
        code.append("    def build(self):")
        return code

    def add_codes(self, indent, codes):
        if isinstance(codes, _string_types):
            codes = codes.strip().split("\n")
        if not isinstance(codes, list):
            raise Exception("Unexpected error!")
        for code in codes:
            self.body_code += (self.tab * indent) + code + "\n"

    def run(self):
        node = self.graph.tf_graph.node[0]
        self.add_codes(0, self.header_code)

        self.save_var_set = set()

        # filter branch nodes, like 'split:1'
        translate_nodes = []
        for node in self.graph.topological_sort:
            if node.count(':') == 0:
                translate_nodes.append(node)

        # check if exists unsupported OPs in model
        if not self.check_op(translate_nodes):
            return

        # ref_name.txt record relationship between
        # paddle value name and tensorflow value name
        ref_name_recorder = open(self.save_dir + "/ref_name.info", 'w')

        total_nodes_num = len(translate_nodes)
        translated_nodes_count = 1
        for node in translate_nodes:
            logging.info("TotalNum:{},TraslatedNum:{},CurrentNode:{}".format(
                total_nodes_num, translated_nodes_count, node))
            current_node = self.graph.get_node(node)
            ref_name_recorder.write("{}\t{}\n".format(
                current_node.layer_name, current_node.output_name))
            translated_nodes_count += 1

            # skip isolated nodes
            if len(current_node.inputs) == 0 and len(
                    current_node.outputs) == 0:
                continue

            op = current_node.layer_type
            if hasattr(self, "emit_" + op):
                func = getattr(self, "emit_" + op)
                func(current_node)
            else:
                raise Exception("Unknow node op: {}".format(op))
        ref_name_recorder.close()

        # merge all the generated python codes
        for node in translate_nodes:
            codes = self.graph.get_node(node).code.gen_codes()
            self.add_codes(2, codes)

        # add return value codes
        outs = []
        for node in self.outputs:
            outs.append(self.graph.get_node(node).output_name)
            self.add_codes(
                2, "# {} : {}".format(
                    self.graph.get_node(node).output_name,
                    self.graph.get_node(node).layer_name))
        input_code = "self.inputs = {}".format([str(s) for s in self.inputs])
        output_code = "self.outputs = [{}]".format(", ".join(outs))
        self.add_codes(2, input_code)
        self.add_codes(2, output_code)

        # write python code to file "my_model.py"
        filew = open(self.save_dir + "/mymodel.py", 'w')
        filew.write(self.body_code)
        filew.close()

        # file "save_var.list" records name of dumped variables
        filew = open(self.save_dir + "/save_var.list", 'w')
        for var in self.save_var_set:
            filew.write(var + '\n')
        filew.close()

        logging.info("Model translated!")
        return self.body_code

    def emit_placeholder(self, node):
        shape = list(self.infer.get_tensor_shape(node.layer))

        self.inputs_sample_data[node.layer_name] = []
        if shape[0] < 0 or shape[0] is None:
            self.batch_node = node
            for i in range(1, 4):
                sample_data = numpy.random.random_sample([i] + shape[1:])
                self.inputs_sample_data[node.layer_name].append(sample_data)
        else:
            for i in range(1, 4):
                sample_data = numpy.random.random_sample(shape)
                self.inputs_sample_data[node.layer_name].append(sample_data)

        if node.data_format == NHWC and len(shape) == 4:
            shape = [shape[0], shape[3], shape[1], shape[2]]

        param_attr = {
            "name": "\'{}\'".format(node.ref_name),
            "shape": shape,
            "dtype": "\'{}\'".format(node.dtype),
            "append_batch_size": False
        }
        node.code.add_layer("data", None, node.output_name, param_attr)

    def emit_const(self, node):
        value = self.infer.get_const_tensor_value(node.layer)
        shape = list(value.shape)

        try:
            dtype = node.dtype
        except:
            return []

        node.code.add_str("#{} {} {}".format(node.layer_name, node.ref_name,
                                             value.shape))
        if value.size == 1:
            param_attr = {
                "shape": [1],
                "value": value.flatten()[0],
                "dtype": "\'{}\'".format(dtype),
            }
            node.code.add_layer("fill_constant", None, node.output_name,
                                param_attr)
        else:
            param_attr = {
                "shape": shape,
                "name": "\'{}\'".format(node.ref_name),
                "dtype": "\'{}\'".format(dtype)
            }
            if node.dtype.startswith('int'):
                param_attr["default_initializer"] = \
                "fluid.initializer.Constant(0)"
            node.code.add_layer("create_parameter", None, node.output_name,
                                param_attr)
            self.export_weights(value, node.ref_name, self.save_dir)

    def emit_conv2d(self, node):
        data = node.inputs[0]
        kernel = node.inputs[1]

        if len(kernel.outputs) == 1:
            kernel.code.clear()

        padding_mode = node.get_attr("padding")
        strides = node.get_attr("strides")[2:4]
        k_shape = list(self.infer.get_tensor_shape(kernel.layer))
        input_shape = list(self.infer.get_tensor_shape(data.layer))
        input_h, input_w = input_shape[2:4]
        k_h, k_w, channel, kernel_num = k_shape
        if node.data_format == NHWC:
            input_h, input_w = input_shape[1:3]
            strides = node.get_attr("strides")[1:3]

        if kernel.layer_name in self.weights:
            weight = self.weights[kernel.layer_name]
            self.weights[kernel.layer_name] = numpy.transpose(
                weight, (3, 2, 0, 1))
            self.export_weights(self.weights[kernel.layer_name],
                                kernel.ref_name, self.save_dir)

        conv2d_param = {
            "num_filters": kernel_num,
            "filter_size": [k_h, k_w],
            "stride": strides,
            "param_attr": "\'{}\'".format(kernel.ref_name),
            "bias_attr": False
        }

        if padding_mode == SAME:
            pad_h = self.compute_padding_size(input_h, k_h, strides[0])
            pad_w = self.compute_padding_size(input_w, k_w, strides[1])
            if len(set(pad_h)) == 1 and len(set(pad_w)) == 1:
                conv2d_param["padding"] = [pad_h[0], pad_w[0]]
                node.code.add_layer("conv2d", data.ref_name, node.output_name,
                                    conv2d_param)
            else:
                pad_param = {"paddings": pad_h + pad_w}
                node.code.add_layer("pad2d", data.ref_name, node.output_name,
                                    pad_param)
                node.code.add_layer("conv2d", node.output_name,
                                    node.output_name, conv2d_param)
        else:
            node.code.add_layer("conv2d", data.ref_name, node.output_name,
                                conv2d_param)

    def emit_variablev2(self, node):
        shape = list(self.infer.get_tensor_shape(node.layer))

        node.code.add_str("# variable[{}]:\t{}".format(node.output_name,
                                                       node.layer_name))

        if node.layer_name in self.weights:
            self.export_weights(self.weights[node.layer_name], node.ref_name,
                                self.save_dir)

        param_attr = {
            "name": "\'{}\'".format(node.ref_name),
            "shape": shape,
            "dtype": "\'{}\'".format(node.dtype)
        }
        if node.dtype.startswith('int'):
            param_attr["default_initializer"] = "fluid.initializer.Constant(0)"
        node.code.add_layer("create_parameter", None, node.output_name,
                            param_attr)

    def emit_biasadd(self, node):
        data = node.inputs[0]
        bias = node.inputs[1]

        if bias.layer_name in self.weights:
            self.export_weights(self.weights[bias.layer_name], bias.ref_name,
                                self.save_dir)

        self.emit_variablev2(bias)
        param_attr = {"x": data.ref_name, "y": bias.ref_name, "axis": 1}
        node.code.add_layer("elementwise_add", None, node.output_name,
                            param_attr)

    def emit_relu(self, node):
        data = node.inputs[0]
        node.code.add_layer("relu", data.ref_name, node.output_name)

    def emit_maxpool(self, node):
        data = node.inputs[0]
        padding_mode = node.get_attr("padding")
        input_shape = list(self.infer.get_tensor_shape(node.layer))
        input_h, input_w = input_shape[2:4]
        strides = node.get_attr("strides")[2:4]
        pool_size = node.get_attr("ksize")[2:4]
        if node.data_format == NHWC:
            input_h, input_w = input_shape[1:3]
            strides = node.get_attr("strides")[1:3]
            pool_size = node.get_attr("ksize")[1:3]

        pool_param = {
            "pool_size": pool_size,
            "pool_type": "\'max\'",
            "pool_stride": strides
        }

        if padding_mode == SAME:
            pad_h = self.compute_padding_size(input_h, pool_size[0],
                                              strides[0])
            pad_w = self.compute_padding_size(input_w, pool_size[1],
                                              strides[1])
            pad_right = pad_w[0] + pad_w[1]
            pad_bottom = pad_h[0] + pad_h[1]
            padding = [0, pad_right, 0, pad_bottom]
            pad_param = {"paddings": padding}
            node.code.add_layer("pad2d", data.ref_name, node.output_name,
                                pad_param)
            node.code.add_layer("pool2d", node.output_name, node.output_name,
                                pool_param)
        else:
            node.code.add_layer("pool2d", data.ref_name, node.output_name,
                                pool_param)

    def emit_squeeze(self, node):
        data = node.inputs[0]
        axis = node.get_attr("squeeze_dims")
        input_shape_len = data.shape_dim_size
        if node.data_format == NHWC and input_shape_len == 4:
            for i in range(0, len(axis)):
                if axis[i] > 0:
                    axis[i] = (axis[i] + 1) % 4 + int((axis[i] + 1) / 4)
        param_attr = {"axes": axis}
        node.code.add_layer("squeeze", data.ref_name, node.output_name,
                            param_attr)

    def emit_add(self, node):
        return self.elementwise(node, "add")

    def emit_mean(self, node):
        data = node.inputs[0]
        reduce_idx = node.inputs[1]
        reduce_idx.code.clear()
        idxs = list(
            self.infer.get_const_tensor_value(reduce_idx.layer).flatten())
        data_shape_len = data.shape_dim_size
        keep_dims = node.layer.attr['keep_dims'].b
        if node.data_format == NHWC and data_shape_len == 4:
            for i in range(len(idxs)):
                if idxs[i] > 0:
                    idxs[i] = (idxs[i] + 1) % 4 + int((idxs[i] + 1) / 4)
        param_attr = {"dim": list(idxs), "keep_dim": keep_dims}
        node.code.add_layer("reduce_mean", data.ref_name, node.output_name,
                            param_attr)

    def emit_fusedbatchnorm(self, node):
        data = node.inputs[0]
        gamma = node.inputs[1]
        beta = node.inputs[2]
        moving_mean = node.inputs[3]
        moving_variance = node.inputs[4]
        if len(gamma.outputs) == 1:
            gamma.code.clear()
        if len(beta.outputs) == 1:
            beta.code.clear()
        if len(moving_mean.outputs) == 1:
            moving_mean.code.clear()
        if len(moving_variance.outputs) == 1:
            moving_variance.code.clear()

        epsilon = round(node.get_attr('epsilon'), 6)
        is_training = node.get_attr('is_training')

        if gamma.layer_name in self.weights:
            self.export_weights(self.weights[gamma.layer_name], gamma.ref_name,
                                self.save_dir)
        if beta.layer_name in self.weights:
            self.export_weights(self.weights[beta.layer_name], beta.ref_name,
                                self.save_dir)
        if moving_mean.layer_name in self.weights:
            self.export_weights(self.weights[moving_mean.layer_name],
                                moving_mean.ref_name, self.save_dir)
        if moving_variance.layer_name in self.weights:
            self.export_weights(self.weights[moving_variance.layer_name],
                                moving_variance.ref_name, self.save_dir)

        param_attr = {
            "epsilon": epsilon,
            "param_attr": "\'{}\'".format(gamma.ref_name),
            "bias_attr": "\'{}\'".format(beta.ref_name),
            "moving_mean_name": "\'{}\'".format(moving_mean.ref_name),
            "moving_variance_name": "\'{}\'".format(moving_variance.ref_name),
            "is_test": not is_training
        }
        node.code.add_layer("batch_norm", data.ref_name, node.output_name,
                            param_attr)

    def emit_concatv2(self, node):
        input_shape_len = node.inputs[0].shape_dim_size
        axis = node.inputs[-1]
        axis.code.clear()
        axis = self.infer.get_const_tensor_value(axis.layer)
        if axis < 0:
            axis = input_shape_len + axis
        if node.data_format == NHWC and input_shape_len == 4:
            if axis > 0:
                axis = (axis + 1) % 4 + int((axis + 1) / 4)
        num_tensor = len(node.inputs) - 1
        input_list = [input.ref_name for input in node.inputs[:num_tensor]]
        input_list = "[{}]".format(", ".join(input_list))
        param_attr = {"axis": axis}
        node.code.add_layer("concat", input_list, node.output_name, param_attr)

    def emit_avgpool(self, node):
        data = node.inputs[0]
        padding_mode = node.get_attr("padding")
        input_shape = list(self.infer.get_tensor_shape(data.layer))
        strides = node.get_attr("strides")[2:4]
        pool_size = node.get_attr("ksize")[2:4]
        input_h, input_w = input_shape[2:4]

        if node.data_format == NHWC:
            strides = node.get_attr("strides")[1:3]
            pool_size = node.get_attr("ksize")[1:3]
            input_h, input_w = input_shape[1:3]

        param_attr = {
            "pool_size": pool_size,
            "pool_stride": strides,
            "pool_type": "\'avg\'"
        }

        if padding_mode == SAME:
            pad_h = self.compute_padding_size(input_h, pool_size[0],
                                              strides[0])
            pad_w = self.compute_padding_size(input_w, pool_size[1],
                                              strides[0])
            if len(set(pad_h)) == 1 and len(set(pad_w)) == 1:
                padding = [pad_h[0], pad_w[0]]
                param_attr["pool_padding"] = padding
            else:
                pad_param = {"paddings": pad_h + pad_w}
                node.code.add_layer("pad2d", data.ref_name, node.output_name,
                                    pad_param)
                node.code.add_layer("pool2d", node.output_name,
                                    node.output_name, param_attr)
        node.code.add_layer("pool2d", data.ref_name, node.output_name,
                            param_attr)

    def emit_rsqrt(self, node):
        data = node.inputs[0]
        pow_param = {"factor": -1.0}
        node.code.add_layer("sqrt", data.ref_name, node.output_name)
        node.code.add_layer("pow", node.output_name, node.output_name,
                            pow_param)

    def emit_mul(self, node):
        return self.elementwise(node, "mul")

    def emit_sub(self, node):
        data1 = node.inputs[0]
        data2 = node.inputs[1]
        axis = self.get_axis(data1, data2)
        data1_shape = list(self.infer.get_tensor_shape(data1.layer))
        data2_shape = list(self.infer.get_tensor_shape(data2.layer))
        param_attr = {"x": data1.ref_name, "y": data2.ref_name, "axis": axis}
        if len(data1_shape) == 4 and len(data2_shape) == 4 \
            and node.data_format == NHWC:
            if data1_shape[-1] != data2_shape[-1]:
                node.code.add_layer("transpose", data1.ref_name, "temp1",
                                    {"perm": [0, 2, 3, 1]})
                node.code.add_layer("transpose", data2.ref_name, "temp2",
                                    {"perm": [0, 2, 3, 1]})
                param_attr = {"x": "temp1", "y": "temp2", "axis": -1}
                node.code.add_layer("elementwise_sub", None, node.output_name,
                                    param_attr)
                node.code.add_layer("transpose", node.output_name,
                                    node.output_name, {"perm": [0, 3, 1, 2]})
        else:
            node.code.add_layer("elementwise_sub", None, node.output_name,
                                param_attr)

    def emit_shape(self, node):
        data = node.inputs[0]
        input_shape_len = data.shape_dim_size
        if input_shape_len == 4 and node.data_format == NHWC:
            param = {"perm": [0, 2, 3, 1]}
            node.code.add_layer("transpose", data.ref_name, node.output_name,
                                param)
            node.code.add_layer("shape", node.output_name, node.output_name)
        else:
            node.code.add_layer("shape", data.ref_name, node.output_name)
        param = {"dtype": "\'int32\'"}
        node.code.add_layer("cast", node.output_name, node.output_name, param)

    def emit_pad(self, node):
        data = node.inputs[0]
        padding = node.inputs[1]
        padding.code.clear()
        padding = padding.layer.attr['value'].tensor
        padding = tensor_util.MakeNdarray(padding).astype('int32')
        if node.data_format == NHWC and padding.shape[0] == 4:
            padding = padding[[0, 3, 1, 2]]
        param_attr = {"paddings": list(padding.flatten())}
        node.code.add_layer("pad", data.ref_name, node.output_name, param_attr)

    def emit_stridedslice(self, node):
        data = node.inputs[0]
        begin = node.inputs[1]
        end = node.inputs[2]
        strides = node.inputs[3]
        begin.code.clear()
        end.code.clear()
        strides.code.clear()

        begin = list(self.infer.get_const_tensor_value(begin.layer).flatten())
        end = list(self.infer.get_const_tensor_value(end.layer).flatten())
        strides = list(
            self.infer.get_const_tensor_value(strides.layer).flatten())

        for i in range(len(strides)):
            assert strides[i] == 1
        param_attr = {"axes": range(len(begin)), "starts": begin, "ends": end}
        node.code.add_layer("slice", data.ref_name, node.output_name,
                            param_attr)

    def emit_resizenearestneighbor(self, node):
        data = node.inputs[0]
        resize_shape = node.inputs[1]
        resize_shape.code.clear()
        align_corners = node.get_attr('align_corners')

        resize_shape = list(self.infer.get_shape_tensor(resize_shape.layer))
        param_attr = {
            "align_corners": align_corners,
            "out_shape": resize_shape
        }
        node.code.add_layer("resize_nearest", data.ref_name, node.output_name,
                            param_attr)

    def emit_maximum(self, node):
        return self.elementwise(node, "max")

    def emit_minimum(self, node):
        return self.elementwise(node, "min")

    def emit_sigmoid(self, node):
        data = node.inputs[0]
        node.code.add_layer("sigmoid", data.ref_name, node.output_name)

    def emit_pack(self, node):
        inputs = [input.ref_name for input in node.inputs]
        inputs = "[{}]".format(", ".join(inputs))
        node.code.add_layer("stack", inputs, node.output_name)

    def emit_reshape(self, node):
        data = node.inputs[0]
        shape = node.inputs[1]
        input_shape_len = data.shape_dim_size
        output_shape = list(self.infer.get_tensor_shape(node.layer))

        shape = self.infer.get_shape_tensor(shape.layer, output_shape)

        reshape_param = {"shape": list(shape)}
        if node.data_format == NHWC and input_shape_len == 4:
            param_attr = {"perm": [0, 2, 3, 1]}
            node.code.add_layer("transpose", data.ref_name, node.output_name,
                                param_attr)
            node.code.add_layer("reshape", node.output_name, node.output_name,
                                reshape_param)
            if len(shape) == 4:
                param_attr = {"perm": [0, 3, 1, 2]}
                node.code.add_layer("transpose", node.output_name,
                                    node.output_name, param_attr)
        else:
            node.code.add_layer("reshape", data.ref_name, node.output_name,
                                reshape_param)

    def emit_conv2dbackpropinput(self, node):
        output_shape = node.inputs[0]
        kernel = node.inputs[1]
        data = node.inputs[2]
        if len(kernel.outputs) == 1:
            kernel.code.clear()
        output_shape.code.clear()
        padding_mode = node.get_attr("padding")
        strides = node.get_attr("strides")[2:4]
        k_shape = self.infer.get_tensor_shape(kernel.layer)
        k_h, k_w, k_num, channel = k_shape
        if node.data_format == NHWC:
            strides = node.get_attr("strides")[1:3]

        padding = [0, 0]
        if padding_mode == SAME:
            padding = [int(val) for val in [(k_h - strides[0]) / 2, \
                (k_w - strides[1]) / 2]]

        if kernel.layer_name in self.weights:
            weight = self.weights[kernel.layer_name]
            self.weights[kernel.layer_name] = numpy.transpose(
                weight, (3, 2, 0, 1))
            self.export_weights(self.weights[kernel.layer_name],
                                kernel.ref_name, self.save_dir)

        output_shape = list(self.infer.get_shape_tensor(output_shape.layer))
        if node.data_format == NHWC and len(output_shape) == 4:
            output_shape = [
                output_shape[0], output_shape[3], output_shape[1],
                output_shape[2]
            ]

        param_attr = {
            "num_filters": k_num,
            "filter_size": [k_h, k_w],
            "padding": padding,
            "stride": strides,
            "param_attr": "\'{}\'".format(kernel.ref_name),
            "bias_attr": False
        }
        node.code.add_layer("conv2d_transpose", data.ref_name,
                            node.output_name, param_attr)
        if padding_mode == SAME:
            param_attr = {"shape": list(output_shape)}
            node.code.add_layer("crop", node.output_name, node.output_name,
                                param_attr)

    def emit_depthwiseconv2dnative(self, node):
        data = node.inputs[0]
        kernel = node.inputs[1]
        if len(kernel.outputs) == 1:
            kernel.code.clear()

        padding_mode = node.get_attr("padding")
        strides = node.get_attr("strides")[2:4]
        k_shape = self.infer.get_tensor_shape(kernel.layer)
        input_shape = self.infer.get_tensor_shape(data.layer)
        input_h, input_w = input_shape[2:4]
        k_h, k_w, in_channels, channel_multiplier = k_shape
        if node.data_format == NHWC:
            strides = node.get_attr("strides")[1:3]
            input_h, input_w = input_shape[1:3]
        groups = channel_multiplier * in_channels

        if kernel.layer_name in self.weights:
            weight = self.weights[kernel.layer_name]
            self.weights[kernel.layer_name] = numpy.transpose(
                weight, (2, 3, 0, 1))
            self.export_weights(self.weights[kernel.layer_name],
                                kernel.ref_name, self.save_dir)
        conv_param = {
            "num_filters": in_channels,
            "filter_size": [k_h, k_w],
            "stride": strides,
            "groups": groups,
            "param_attr": "\'{}\'".format(kernel.ref_name),
            "bias_attr": False
        }
        if padding_mode == SAME:
            pad_h = self.compute_padding_size(input_h, k_h, strides[0])
            pad_w = self.compute_padding_size(input_w, k_w, strides[1])
            if len(set(pad_h)) == 1 and len(set(pad_w)) == 1:
                padding = [pad_h[0], pad_w[0]]
                conv_param["padding"] = padding
                node.code.add_layer("conv2d", data.ref_name, node.output_name,
                                    conv_param)
            else:
                pad_param = {"paddings": pad_h + pad_w}
                node.code.add_layer("pad2d", data.ref_name, node.output_name,
                                    pad_param)
                node.code.add_layer("conv2d", node.output_name,
                                    node.output_name, conv_param)
        else:
            node.code.add_layer("conv2d", data.ref_name, node.output_name,
                                conv_param)

    def emit_softmax(self, node):
        data = node.inputs[0]
        node.code.add_layer("softmax", data.ref_name, node.output_name)

    def emit_matmul(self, node):
        data0 = node.inputs[0]
        data1 = node.inputs[1]
        transpose_a = node.get_attr('transpose_a')
        transpose_b = node.get_attr('transpose_b')
        param_attr = {
            "x": data0.ref_name,
            "y": data1.ref_name,
            "transpose_x": transpose_a,
            "transpose_y": transpose_b
        }
        node.code.add_layer("matmul", None, node.output_name, param_attr)

    def emit_transpose(self, node):
        data = node.inputs[0]
        perm = node.inputs[1]
        perm.code.clear()
        perm = list(self.infer.get_shape_tensor(perm.layer))
        if node.data_format == NHWC and len(perm) == 4:
            if perm == [0, 3, 1, 2]:
                self.graph.set_data_format(node, NCHW)
                node.code.add_str("{} = {}".format(node.output_name,
                                                   data.ref_name))
            else:
                raise Exception("Unexpected situation in OP transpose")
        elif node.data_format == NCHW and len(perm) == 4:
            if perm == [0, 2, 3, 1]:
                self.graph.set_data_format(node, NHWC)
                node.code.add_str("{} = {}".format(node.output_name,
                                                   data.ref_name))
            else:
                raise Exception("Unexpected situation in OP transpose")
        else:
            param_attr = {"perm": perm}
            node.code.add_layer("transpose", data.ref_name, node.output_name,
                                param_attr)

    def emit_randomuniform(self, node):
        shape = node.inputs[0]
        shape = self.infer.get_shape_tensor(shape.layer)
        if node.data_format == NHWC and len(shape) == 4:
            shape = shape[[0, 3, 1, 2]]
        batch_index = list(numpy.argwhere(shape < 0).flatten())
        shape = list(shape)
        param_attr = {
            "shape": shape,
            "dtype": "\'float32\'",
            "min": 0.00001,
            "max": 0.99999
        }
        if len(batch_index) > 1:
            raise Exception("More than one dimension value less than zero")
        if len(batch_index) == 0:
            node.code.add_layer("uniform_random", None, node.output_name,
                                param_attr)
        else:
            param_attr["input_dim_idx"] = batch_index[0]
            node.code.add_layer("uniform_random_batch_size_like",
                                self.batch_node.ref_name, node.output_name,
                                param_attr)

    def emit_floor(self, node):
        data = node.inputs[0]
        node.code.add_layer("floor", data.ref_name, node.output_name)

    def emit_exp(self, node):
        data = node.inputs[0]
        node.code.add_layer("exp", data.ref_name, node.output_name)

    def emit_floordiv(self, node):
        self.emit_div(node)
        param = {"dtype": "\'float32\'"}
        node.code.add_layer("cast", node.output_name, node.output_name, param)
        node.code.add_layer("floor", node.output_name, node.output_name)

    def emit_div(self, node):
        data1 = node.inputs[0]
        data2 = node.inputs[1]
        axis = self.get_axis(data1, data2)
        data1_shape = self.infer.get_tensor_shape(data1.layer)
        data2_shape = self.infer.get_tensor_shape(data2.layer)
        div_param = {"x": data1.ref_name, "y": data2.ref_name, "axis": axis}
        if len(data1_shape) == 4 and len(data2_shape) == 4 \
            and node.data_format == NHWC:
            if data1_shape[-1] != data2_shape[-1]:
                perm = {"perm": [0, 2, 3, 1]}
                node.code.add_layer("transpose", data1.ref_name, "temp1", perm)
                node.code.add_layer("transpose", data2.ref_name, "temp2", perm)
                div_param["x"] = "temp1"
                div_param["y"] = "temp2"
                div_param["axis"] = -1
        node.code.add_layer("elementwise_div", None, node.output_name,
                            div_param)

    def emit_realdiv(self, node):
        return self.emit_div(node)

    def emit_slice(self, node):
        data = node.inputs[0]
        begin = node.inputs[1]
        size = node.inputs[2]
        begin.code.clear()
        size.code.clear()
        begin = list(self.infer.get_shape_tensor(begin.layer))
        size = list(self.infer.get_shape_tensor(size.layer))

        input_shape = self.infer.get_tensor_shape(data.layer)
        if len(numpy.argwhere(input_shape < 0).flatten()) > 1:
            input_shape = list(self.infer.get_tensor_shape(data.layer))

        assert len(begin) == len(input_shape) and len(size) == len(input_shape)

        if node.data_format == NHWC and len(input_shape) == 4:
            begin = [begin[0], begin[3], begin[1], begin[2]]
            size = [size[0], size[3], size[1], size[2]]
            input_shape = [
                input_shape[0], input_shape[3], input_shape[1], input_shape[2]
            ]

        for i in range(len(size)):
            if size[i] < 0:
                size[i] = input_shape[i] - begin[i]
        param_attr = {"shape": size, "offsets": begin}
        node.code.add_layer("crop", data.ref_name, node.output_name,
                            param_attr)

    def emit_sum(self, node):
        data = node.inputs[0]
        reduce_idx = node.inputs[1]
        reduce_idx.code.clear()
        idxs = tensor_util.MakeNdarray(
            reduce_idx.layer.attr['value'].tensor).astype('int32').flatten()
        data_shape_len = data.shape_dim_size
        keep_dims = node.layer.attr['keep_dims'].b
        if node.data_format == NHWC and data_shape_len == 4:
            for i in range(idxs.shape[0]):
                if idxs[i] > 0:
                    idxs[i] = (idxs[i] + 1) % 4 + int((idxs[i] + 1) / 4)
        param = {"dim": list(idxs), "keep_dim": keep_dims}
        node.code.add_layer("reduce_sum", data.ref_name, node.output_name,
                            param)

    def emit_max(self, node):
        data = node.inputs[0]
        reduce_idx = node.inputs[1]
        reduce_idx.code.clear()
        idxs = tensor_util.MakeNdarray(
            reduce_idx.layer.attr['value'].tensor).astype('int32').flatten()
        data_shape_len = data.shape_dim_size
        keep_dims = node.layer.attr['keep_dims'].b
        if node.data_format == NHWC and data_shape_len == 4:
            for i in range(idxs.shape[0]):
                if idxs[i] > 0:
                    idxs[i] = (idxs[i] + 1) % 4 + int((idxs[i] + 1) / 4)
        param = {"dim": list(idxs), "keep_dim": keep_dims}
        node.code.add_layer("reduce_max", data.ref_name, node.output_name,
                            param)

    def emit_fill(self, node):
        shape = node.inputs[0]
        shape.code.clear()
        value = node.inputs[1]
        value.code.clear()

        shape = list(self.infer.get_shape_tensor(shape.layer))
        value = list(self.infer.get_const_tensor_value(value.layer).flatten())
        assert len(value) == 1
        value = value[0]

        if node.data_format == NHWC and len(shape) == 4:
            shape = [shape[0], shape[3], shape[1], shape[2]]

        param = {
            "shape": shape,
            "dtype": "\'{}\'".format(value.dtype),
            "value": value
        }
        if shape[0] < 0:
            node.code.add_layer("fill_constant_batch_size_like",
                                self.batch_node.ref_name, node.output_name,
                                param)
        else:
            node.code.add_layer("fill_constant", None, node.output_name, param)

    def emit_range(self, node):
        start = node.inputs[0]
        end = node.inputs[1]
        delta = node.inputs[2]
        start.code.clear()
        end.code.clear()
        delta.code.clear()

        start = self.infer.get_const_tensor_value(start.layer)
        end = self.infer.get_const_tensor_value(end.layer)
        delta = self.infer.get_const_tensor_value(delta.layer)
        np_code = "np_array = numpy.arange({}, {}, {}).astype(\'{}\')".format(
            start, end, delta, delta.dtype)
        node.code.add_str(np_code)
        node.code.add_layer("assign", "np_array", node.output_name)

    def emit_tile(self, node):
        data = node.inputs[0]
        expand_times = node.inputs[1]
        expand_times.code.clear()
        expand_times = list(
            self.infer.get_const_tensor_value(expand_times.layer))
        param = {"expand_times": expand_times}
        node.code.add_layer("expand", data.ref_name, node.output_name, param)

    def emit_splitv(self, node):
        data = node.inputs[0]
        num_sections = node.inputs[1]
        num_sections.code.clear()
        split_dim = node.inputs[2]
        split_dim.code.clear()
        num_sections = self.infer.get_const_tensor_value(num_sections.layer)
        split_dim = self.infer.get_const_tensor_value(split_dim.layer)
        input_shape = self.infer.get_tensor_shape(data.layer)
        if split_dim < 0:
            split_dim += len(input_shape)

        index = numpy.argwhere(num_sections < 0).flatten()
        if index.shape[0] > 1:
            raise Exception("More than one dimension less than 0")
        if index.shape[0] == 1:
            num_sections[index[0]] = input_shape[split_dim] - numpy.sum(
                num_sections) + num_sections[index[0]]
        param = {"num_or_sections": list(num_sections), "dim": split_dim}
        node.code.add_layer("split", data.ref_name, node.output_name, param)

    def emit_expanddims(self, node):
        data = node.inputs[0]
        dim = node.inputs[1]
        dim.code.clear()
        dim = self.infer.get_const_tensor_value(dim.layer)
        param = {"axes":[dim]}
        node.code.add_layer("unsqueeze", data.ref_name, node.output_name, param)
