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
from six import string_types as _string_types
import framework_pb2 as framework
import math
import struct
import numpy
import logging
import sys


class PaddleEmitter(object):
    skip_op = set(['variablev2', 'identity'])
    skip_op = set()
    dtype_map = {1: "float32", 3: "int32", 9: "int64"}

    def __init__(self, parser, save_dir):
        self.graph = parser.tf_graph
        self.weights = parser.weights
        self.save_dir = save_dir
        self.body_code = ""
        self.tab = " " * 4

    @staticmethod
    def tensor_shape_to_list(shapes):
        if isinstance(shapes, attr_value_pb2.AttrValue):
            return [dim.size for dim in shapes.shape.dim]
        else:
            ret = []
            for shape in shapes:
                this_one = [dim.size for dim in shape.dim]
                ret.append(this_one)
            return ret

    @staticmethod
    def compute_padding_size(in_size, filter_size, stride):
        new_size = int(math.ceil(in_size * 1.0 / stride))
        pad_size = (new_size - 1) * stride + filter_size - in_size
        return pad_size

    def check_op(self, node_name_list):
        uncovered_ops = set()
        for name in node_name_list:
            node = self.graph.get_node(name)
            if len(node.inputs) == 0 and len(node.outputs) == 0:
                continue
            if node.layer_type in self.skip_op:
                continue
            if not hasattr(self, "emit_" + node.layer_type):
                uncovered_ops.add(node.layer_type)
        if len(uncovered_ops) > 0:
            sys.stderr.write("Still {} tensorflow OP are not supported\n".format(len(uncovered_ops)))
            for op in uncovered_ops:
                sys.stderr.write("Unsupported OP: {}\n".format(op))
            sys.exit(0)

    def get_axis(self, node1, node2):
        shape1 = self.tensor_shape_to_list(node1.get_attr('_output_shapes'))[0]
        shape2 = self.tensor_shape_to_list(node2.get_attr('_output_shapes'))[0]
        if len(shape1) == 4 and len(
                shape2) == 1 and node1.data_format == "NHWC":
            axis = 1
        elif len(shape2) == 4 and len(
                shape1) == 1 and node2.data_format == "NHWC":
            axis = 1
        else:
            axis = -1
        return axis

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
        tensor_stream = ""
        for i in range(0, tensor_size):
            tensor_stream += struct.pack(struct_write_format[str(weight.dtype)], weight[i])
        filew.write(tensor_stream)
        filew.close()

    @property
    def header_code(self):
        code = list()
        code.append("import paddle.fluid.layers as layers")
        code.append("import paddle.fluid as fluid")
        code.append("")
        code.append("def KitModel():")
        return code

    def add_codes(self, indent, codes):
        if isinstance(codes, _string_types):
            codes = codes.strip().split("\n")
        for code in codes:
            self.body_code += (self.tab * indent) + code + "\n"

    def run(self):
        node = self.graph.tf_graph.node[0]
        self.add_codes(0, self.header_code)

        self.save_var_set = set()

        self.check_op(self.graph.topological_sort)
        ref_name_recorder = open(self.save_dir + "/ref_name.txt", 'w')
        total_nodes_num = len(self.graph.topological_sort)
        translated_nodes_count = 1
        sys.stderr.write("\nModel Translating......\n")
        sys.stderr.write("Start to translate all the nodes(Total_num:{})\n".
                         format(total_nodes_num))
        for node in self.graph.topological_sort:
            sys.stderr.write(
                "\rTranslated nodes num: {}, Current node: {}".format(
                    translated_nodes_count, node))
            sys.stderr.flush()
            translated_nodes_count += 1
            current_node = self.graph.get_node(node)
            ref_name_recorder.write("{}\t{}\n".format(
                current_node.layer_name, current_node.output_name))

            # skip isolated nodes
            if len(current_node.inputs) == 0 and len(
                    current_node.outputs) == 0:
                continue

            op = current_node.layer_type
            if op in self.skip_op:
                continue
            if hasattr(self, "emit_" + op):
                func = getattr(self, "emit_" + op)
                codes = func(current_node)
                if not isinstance(codes, list):
                    codes = [codes]
                self.graph.get_node(node).codes = codes
            else:
                raise Exception("Unknow node op: {}".format(op))
        ref_name_recorder.close()

        for node in self.graph.topological_sort:
            codes = self.graph.get_node(node).codes
            self.add_codes(1, codes)

        outs = []
        for node in self.graph.output_nodes:
            outs.append(self.graph.get_node(node).output_name)
        self.add_codes(1, "return {}".format(", ".join(outs)))

        filew = open(self.save_dir + "/mymodel.py", 'w')
        filew.write(self.body_code)
        filew.close()
        filew = open(self.save_dir + "/save_var.list", 'w')
        for var in self.save_var_set:
            filew.write(var + '\n')
        filew.close()

        sys.stderr.write("Model translated!\n\n")
        sys.stderr.flush()

        return self.body_code

    def emit_placeholder(self, node):
        shape = self.tensor_shape_to_list(node.get_attr('_output_shapes'))[0]

        if node.data_format == "NHWC" and len(shape) == 4:
            shape = [shape[0], shape[3], shape[1], shape[2]]

        dtype = node.get_attr("dtype")
        if dtype in self.dtype_map:
            dtype = self.dtype_map[dtype]
        else:
            raise Exception("Unknow dtype: {}".format(dtype))

        code = list()
        code.append("# placehoder[{}]:\t{}".format(node.output_name,
                                                   node.layer_name))
        code.append(
            "{} = layers.data(name=\'{}\', shape={}, dtype=\'{}\')".format(
                node.output_name, node.ref_name, shape, dtype))
        return code

    def emit_const(self, node):
        try:
            float_val = node.layer.attr['value'].tensor.float_val
            if len(float_val) == 1:
                code = list()
                code.append("# {} {}".format(node.layer_name, node.ref_name))
                code.append(
                    "{} = layers.fill_constant(shape=[1], value={}, dtype=\'float32\')"
                    .format(node.output_name, float_val[0]))
                return code
            int_val = node.layer.attr['value'].tensor.int_val
            if len(int_val) == 1:
                code = list()
                code.append("# {} {}".format(node.layer_name, node.ref_name))
                code.append(
                    "{} = layers.fill_constant(shape=[1], value={}, dtype=\'int32\')"
                    .format(node.output_name, int_val[0]))
                return code

            node.layer.attr['value'].tensor.tensor_content
            weight = tensor_util.MakeNdarray(node.layer.attr['value'].tensor)
            if len(weight) == 0:
                return []

            shape = list(weight.shape)
            dtype = node.get_attr('dtype')

            if dtype in self.dtype_map:
                dtype = self.dtype_map[dtype]
            else:
                raise Exception("Unknow dtype[{}] of node[{}]".format(
                    dtype, node.layer_name))

            code = list()
            code.append("# {} {}".format(node.layer_name, node.ref_name))
            if dtype.startswith('int'):
                code.append(
                    "{} = layers.create_parameter({}, \'{}\', \'{}\', default_initializer=fluid.initializer.Constant(0))"
                    .format(node.output_name, shape, dtype, node.ref_name))
            else:
                code.append(
                    "{} = layers.create_parameter({}, \'{}\', \'{}\')".format(
                        node.output_name, shape, dtype, node.ref_name))

            self.export_weights(weight, node.ref_name, self.save_dir)

            return code
        except:
            return []

    def emit_conv2d(self, node):
        data = node.inputs[0]
        kernel = node.inputs[1]
        if len(kernel.outputs) == 1:
            kernel.codes = []

        padding_mode = node.get_attr("padding")
        strides = node.get_attr("strides")[1:3]
        k_shape = self.tensor_shape_to_list(
            kernel.get_attr("_output_shapes"))[0]
        input_shape = self.tensor_shape_to_list(
            data.get_attr("_output_shapes"))[0]
        input_h, input_w = input_shape[2:4]
        kernel_num, channel, k_h, k_w = k_shape
        if node.data_format == "NHWC":
            k_h, k_w, channel, kernel_num = k_shape
            input_h, input_w = input_shape[1:3]
        if k_h < strides[0] or k_w < strides[1]:
            raise Exception(
                "Unexpected situation with kernel's height/width less than the corresponding stride"
            )

        if kernel.layer_name in self.weights:
            if node.data_format == "NHWC":
                weight = self.weights[kernel.layer_name]
                self.weights[kernel.layer_name] = numpy.transpose(
                    weight, (3, 2, 0, 1))
            self.export_weights(self.weights[kernel.layer_name],
                                kernel.ref_name, self.save_dir)

        code = list()
        padding = [0, 0]
        if padding_mode == "SAME":
            total_pad_h = self.compute_padding_size(input_h, k_h, strides[0])
            total_pad_w = self.compute_padding_size(input_w, k_w, strides[1])
            if total_pad_h % 2 == 0 and total_pad_w % 2 == 0:
                padding = map(int, [total_pad_h / 2, total_pad_w / 2])
                code.append(
                    "{} = layers.conv2d({}, {}, {}, padding={}, stride={}, param_attr=\'{}\', bias_attr=False)"
                    .format(node.output_name, data.ref_name, kernel_num,
                            [k_h, k_w], padding, strides, kernel.ref_name))
            else:
                padding = [0] * 4
                padding[0] = total_pad_h / 2
                padding[1] = total_pad_h - padding[0]
                padding[2] = total_pad_w / 2
                padding[3] = total_pad_w - padding[2]
                code.append("{} = layers.pad2d({}, {})".format(
                    node.output_name, data.ref_name, padding))
                code.append(
                    "{} = layers.conv2d({}, {}, {}, stride={}, param_attr=\'{}\', bias_attr=False)"
                    .format(node.output_name, node.ref_name, kernel_num,
                            [k_h, k_w], strides, kernel.ref_name))
        else:
            code.append(
                "{} = layers.conv2d({}, {}, {}, stride={}, param_attr=\'{}\', bias_attr=False)"
                .format(node.output_name, data.ref_name, kernel_num,
                        [k_h, k_w], strides, kernel.ref_name))
        return code

    def emit_variablev2(self, node):
        shape = self.tensor_shape_to_list(node.get_attr("_output_shapes"))[0]
        dtype = node.get_attr("dtype")
        if dtype in self.dtype_map:
            dtype = self.dtype_map[dtype]
        else:
            raise Exception("Unknow dtype[{}] of node[{}]".format(
                dtype, node.layer_name))

        code = list()
        code.append("# variable[{}]:\t{}".format(node.output_name,
                                                 node.layer_name))
        if dtype.startswith('int'):
            code.append(
                "{} = layers.create_parameter(name=\'{}\', shape={}, dtype=\'{}\', default_initializer=fluid.initializer.Constant(0))"
                .format(node.output_name, node.ref_name, shape, dtype))
        else:
            code.append(
                "{} = layers.create_parameter(name=\'{}\', shape={}, dtype=\'{}\')"
                .format(node.output_name, node.ref_name, shape, dtype))
        return code

    def emit_biasadd(self, node):
        data = node.inputs[0]
        bias = node.inputs[1]
        axis = self.get_axis(data, bias)

        if bias.layer_name in self.weights:
            self.export_weights(self.weights[bias.layer_name], bias.ref_name,
                                self.save_dir)

        code = list()
        code = code + self.emit_variablev2(bias)
        code.append("{} = layers.elementwise_add({}, {}, axis={})".format(
            node.output_name, data.ref_name, bias.ref_name, axis))
        return code

    def emit_relu(self, node):
        data = node.inputs[0]
        code = "{} = layers.relu({})".format(node.output_name, data.ref_name)
        return code

    def emit_maxpool(self, node):
        data = node.inputs[0]
        padding_mode = node.get_attr("padding")
        strides = node.get_attr("strides")[1:3]
        pool_size = node.get_attr("ksize")[1:3]
        if padding_mode == "SAME":
            pad_right = (pool_size[0] - 1) / 2
            pad_bottom = (pool_size[1] - 1) / 2
            padding = [0, pad_right * 2, 0, pad_bottom * 2]
            code = [
                "pad_net = layers.pad2d({}, paddings={})".format(
                    data.ref_name, padding)
            ]
            code.append("{} = layers.pool2d(pad_net, {}, \'max\', {})".format(
                node.output_name, pool_size, strides))
            return code
        else:
            code = "{} = layers.pool2d({}, {}, \'max\', {})".format(
                node.output_name, data.ref_name, pool_size, strides)
            return code

    def emit_squeeze(self, node):
        data = node.inputs[0]
        axis = node.get_attr("squeeze_dims")
        input_shape = self.tensor_shape_to_list(
            data.get_attr("_output_shapes"))[0]
        if node.data_format == "NHWC" and len(input_shape) == 4:
            for i in range(0, len(axis)):
                if axis[i] == 1:
                    axis[i] = 2
                elif axis[i] == 2:
                    axis[i] = 3
                elif axis[i] == 3:
                    axis[i] = 1
        code = "{} = layers.squeeze({}, {})".format(node.output_name,
                                                    data.ref_name, axis)
        return code

    def emit_add(self, node):
        data1 = node.inputs[0]
        data2 = node.inputs[1]
        axis = self.get_axis(data1, data2)
        code = "{} = layers.elementwise_add({}, {}, axis={})".format(node.output_name, data1.ref_name, data2.ref_name, axis)
        return code

    def emit_mean(self, node):
        data = node.inputs[0]
        reduce_idx = node.inputs[1]
        reduce_idx.codes = []
        idxs = tensor_util.MakeNdarray(reduce_idx.layer.attr['value'].tensor)
        shape = idxs.shape
        assert len(shape) == 1
        data_shape = self.tensor_shape_to_list(
            data.get_attr('_output_shapes'))[0]
        keep_dims = node.layer.attr['keep_dims'].b
        if node.data_format == "NHWC" and len(data_shape) == 4:
            for i in range(0, shape[0]):
                if idxs[i] == 1:
                    idxs[i] = 2
                elif idxs[i] == 2:
                    idxs[i] = 3
                elif idxs[i] == 3:
                    idxs[i] = 1

        code = "{} = layers.reduce_mean({}, {}, keep_dim={})".format(
            node.output_name, data.ref_name, list(idxs), keep_dims)
        return code

    def emit_fusedbatchnorm(self, node):
        data = node.inputs[0]
        gamma = node.inputs[1]
        beta = node.inputs[2]
        moving_mean = node.inputs[3]
        moving_variance = node.inputs[4]
        if len(gamma.outputs) == 1:
            gamma.codes = []
        if len(beta.outputs) == 1:
            beta.codes = []
        if len(moving_mean.outputs) == 1:
            moving_mean.codes = []
        if len(moving_variance.outputs) == 1:
            moving_variance.codes = []

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

        code = "{} = layers.batch_norm({}, epsilon={}, param_attr=\'{}\', bias_attr=\'{}\', moving_mean_name=\'{}\', moving_variance_name=\'{}\', is_test={})".format(
            node.output_name, data.ref_name, epsilon, gamma.ref_name,
            beta.ref_name, moving_mean.ref_name, moving_variance.ref_name,
            not is_training)
        return code

    def emit_concatv2(self, node):
        input_shape = self.tensor_shape_to_list(
            node.inputs[0].get_attr('_output_shapes'))[0]
        axis = node.inputs[-1]
        axis.codes = []
        axis = axis.layer.attr['value'].tensor.int_val[0]
        if node.data_format == "NHWC" and len(input_shape) == 4:
            if axis == 1:
                axis = 2
            elif axis == 2:
                axis = 3
            elif axis == 3:
                axis = 1

        num_tensor = len(node.inputs) - 1
        code = "{} = layers.concat([{}], {})".format(
            node.output_name, ", ".join(
                [input.ref_name for input in node.inputs[:num_tensor]]), axis)
        return code

    def emit_avgpool(self, node):
        data = node.inputs[0]
        padding_mode = node.get_attr("padding")
        strides = node.get_attr("strides")[1:3]
        pool_size = node.get_attr("ksize")[1:3]
        padding = [0, 0]
        if padding_mode == "SAME":
            pad_h = (pool_size[0] - 1) / 2
            pad_w = (pool_size[1] - 1) / 2
            padding = [pad_h, pad_w]
        code = "{} = layers.pool2d({}, {}, \'avg\', {}, {})".format(
            node.output_name, data.ref_name, pool_size, strides, padding)
        return code

    def emit_rsqrt(self, node):
        data = node.inputs[0]
        code = list()
        code.append("sqrt_res = layers.sqrt({})".format(data.ref_name))
        code.append("{} = layers.pow(sqrt_res, -1.0)".format(node.output_name))
        return code

    def emit_mul(self, node):
        data1 = node.inputs[0]
        data2 = node.inputs[1]
        axis = self.get_axis(data1, data2)
        code = "{} = layers.elementwise_mul({}, {}, axis={})".format(
            node.output_name, data1.ref_name, data2.ref_name, axis)
        shape1 = self.tensor_shape_to_list(data1.get_attr('_output_shapes'))[0]
        shape2 = self.tensor_shape_to_list(data2.get_attr('_output_shapes'))[0]
        if len(shape2) > len(shape1):
            code = "{} = layers.elementwise_mul({}, {}, axis={})".format(
                node.output_name, data2.ref_name, data1.ref_name, axis)
        return code

    def emit_sub(self, node):
        data1 = node.inputs[0]
        data2 = node.inputs[1]
        axis = self.get_axis(data1, data2)
        code = "{} = layers.elementwise_sub({}, {}, axis={})".format(
            node.output_name, data1.ref_name, data2.ref_name, axis)
        return code

    def emit_shape(self, node):
        data = node.inputs[0]
        code = "{} = layers.shape({})".format(node.output_name, data.ref_name)
        return code

    def emit_pad(self, node):
        data = node.inputs[0]
        padding = node.inputs[1]
        padding.codes = []
        padding = padding.layer.attr['value'].tensor
        padding = tensor_util.MakeNdarray(padding)
        if node.data_format == "NHWC" and padding.shape[0] == 4:
            padding = padding[[0, 3, 1, 2]]
        code = "{} = layers.pad({}, {})".format(node.output_name,
                                                data.ref_name,
                                                list(padding.flatten()))
        return code

    def emit_stridedslice(self, node):
        data = node.inputs[0]
        begin = node.inputs[1]
        end = node.inputs[2]
        strides = node.inputs[3]
        begin.codes = []
        end.codes = []
        strides.codes = []
        begin = list(tensor_util.MakeNdarray(begin.layer.attr['value'].tensor))
        end = list(tensor_util.MakeNdarray(end.layer.attr['value'].tensor))
        strides = list(
            tensor_util.MakeNdarray(strides.layer.attr['value'].tensor))

        for i in range(len(strides)):
            assert strides[i] == 1

        code = "{} = layers.slice({}, axes={}, starts={}, ends={})".format(
            node.output_name, data.ref_name, [i for i in range(len(begin))],
            begin, end)
        return code

    def emit_resizenearestneighbor(self, node):
        data = node.inputs[0]
        output_shape = node.inputs[1]
        align_corners = node.get_attr('align_corners')

        if output_shape.layer_type == "const":
            output_shape.codes = []
            output_shape = tensor_util.MakeNdarray(
                output_shape.layer.attr['value'].tensor)
            code = "{} = layers.resize_nearest({}, {}, align_corners={}, align_mode=1)".format(
                node.output_name, data.ref_name, list(output_shape),
                align_corners)
        else:
            code = "{} = layers.resize_nearest({}, {}, align_corners={}, align_mode=1)".format(
                node.output_name, data.ref_name, output_shape.ref_name,
                align_corners)
            logging.warn(
                "\tNotice there's RESIZE_NEAREST in translated code, and the code list below:"
            )
            logging.warn("\t\t{}".format(code))
            logging.warn(
                "\tPaddle doesn't support tensor type for output_shape now")
            logging.warn(
                "\tYou need to change \'{}\'(in tf model: \'{}\') to a list with constant value, e.g. [28, 28]. IMPORTANT!!!\n"
                .format(output_shape.ref_name, output_shape.layer_name))
        return code

    def emit_maximum(self, node):
        data1 = node.inputs[0]
        data2 = node.inputs[1]
        axis = self.get_axis(data1, data2)
        code = "{} = layers.elementwise_max({}, {}, axis={})".format(
            node.output_name, data1.ref_name, data2.ref_name, axis)
        return code

    def emit_minimum(self, node):
        data1 = node.inputs[0]
        data2 = node.inputs[1]
        axis = self.get_axis(data1, data2)
        code = "{} = layers.elementwise_min({}, {}, axis={})".format(
            node.output_name, data1.ref_name, data2.ref_name, axis)
        return code

    def emit_sigmoid(self, node):
        data = node.inputs[0]
        code = "{} = layers.sigmoid({})".format(node.output_name,
                                                data.ref_name)
        return code

    def emit_pack(self, node):
        ref_name = [input.ref_name for input in node.inputs]
        code = "{} = layers.stack([{}])".format(node.output_name,
                                                ", ".join(ref_name))
        return code

    def emit_reshape(self, node):
        data = node.inputs[0]
        shape = node.inputs[1]
        if shape.layer_type == "const":
            shape = shape.layer.attr['value'].tensor
            shape = list(tensor_util.MakeNdarray(shape))
            code = "{} = layers.reshape({}, {})".format(
                node.output_name, data.ref_name, shape)
        else:
            code = "{} = layers.reshape({}, {})".format(
                node.output_name, data.ref_name, shape.ref_name)
            logging.warn(
                "\tNotice there's RESHAPE in translated code, and the code list below:"
            )
            logging.warn("\t\t{}".format(code))
            logging.warn(
                "\tPaddle doesn't support tensor type for output_shape now")
            logging.warn(
                "\tYou need to change \'{}\'(in tf model: \'{}\') to a list with constant value, e.g. [28, 28]. IMPORTANT!!!\n"
                .format(shape.ref_name, shape.layer_name))

        return code

    def emit_conv2dbackpropinput(self, node):
        output_shape = node.inputs[0]
        kernel = node.inputs[1]
        data = node.inputs[2]
        if len(kernel.outputs) == 1:
            kernel.codes = []
        padding_mode = node.get_attr("padding")
        strides = node.get_attr("strides")[1:3]
        k_shape = self.tensor_shape_to_list(
            kernel.get_attr("_output_shapes"))[0]

        channel, k_num, k_h, k_w = k_shape
        if node.data_format == "NHWC":
            k_h, k_w, k_num, channel = k_shape

        if strides[0] > k_h or strides[1] > k_w:
            raise Exception(
                "Paddle cannot process the situation now[kernel's height/width less than the corresponding stride]"
            )

        padding = [0, 0]
        if padding_mode == "SAME":
            padding = map(int, [(k_h - strides[0]) / 2,
                                (k_w - strides[1]) / 2])

        if kernel.layer_name in self.weights:
            if node.data_format == "NHWC":
                weight = self.weights[kernel.layer_name]
                self.weights[kernel.layer_name] = numpy.transpose(
                    weight, (3, 2, 0, 1))
                self.export_weights(self.weights[kernel.layer_name],
                                    kernel.ref_name, self.save_dir)

        code = []
        if output_shape.layer_type == "const":
            output_shape.codes = []
            output_shape = tensor_util.MakeNdarray(
                output_shape.layer.attr['value'].tensor)
            if node.data_format == "NHWC" and output_shape[0] == 4:
                output_shape = output_shape[[0, 3, 1, 2]]
            code.append(
                "{} = layers.conv2d_transpose({}, {}, None, {}, {}, {}, param_attr=\'{}\', bias_attr=False)"
                .format(node.output_name, data.ref_name, k_num, [k_h, k_w],
                        padding, strides, kernel.ref_name))
            if padding_mode == "SAME":
                code.append("{} = layers.crop({}, shape={})".format(
                    node.output_name, node.output_name, list(output_shape)))
        else:
            code.append(
                "{} = layers.conv2d_transpose({}, {}, None, {}, 0, {}, param_attr=\'{}\', bias_attr=False)"
                .format(node.output_name, data.ref_name, k_num, [k_h, k_w],
                        strides, kernel.ref_name))
            if padding_mode == "SAME":
                code.append("{} = layers.crop({}, shape={})".format(
                    node.output_name, node.output_name, output_shape.ref_name))

        return code

    def emit_depthwiseconv2dnative(self, node):
        data = node.inputs[0]
        kernel = node.inputs[1]
        if len(kernel.outputs) == 1:
            kernel.codes = []

        padding_mode = node.get_attr("padding")
        strides = node.get_attr("strides")[1:3]
        k_shape = self.tensor_shape_to_list(
            kernel.get_attr("_output_shapes"))[0]
        input_shape = self.tensor_shape_to_list(
            data.get_attr("_output_shapes"))[0]
        input_h, input_w = input_shape[2:4]
        in_channels, channel_multiplier, k_h, k_w = k_shape
        if node.data_format == "NHWC":
            k_h, k_w, in_channels, channel_multiplier = k_shape
            input_h, input_w = input_shape[1:3]
        if k_h < strides[0] or k_w < strides[1]:
            raise Exception(
                "Unexpected situation with kernel's height/width less than the corresponding stride"
            )

        groups = channel_multiplier * in_channels

        if kernel.layer_name in self.weights:
            if node.data_format == "NHWC":
                weight = self.weights[kernel.layer_name]
                self.weights[kernel.layer_name] = numpy.transpose(
                    weight, (2, 3, 0, 1))
            self.export_weights(self.weights[kernel.layer_name],
                                kernel.ref_name, self.save_dir)

        code = list()
        padding = [0, 0]
        if padding_mode == "SAME":
            total_pad_h = self.compute_padding_size(input_h, k_h, strides[0])
            total_pad_w = self.compute_padding_size(input_w, k_w, strides[1])
            if total_pad_h % 2 == 0 and total_pad_w % 2 == 0:
                padding = map(int, [total_pad_h / 2, total_pad_w / 2])
                code.append(
                    "{} = layers.conv2d({}, {}, {}, padding={}, stride={}, param_attr=\'{}\', bias_attr=False, groups={})"
                    .format(node.output_name, data.ref_name, in_channels,
                            [k_h, k_w], padding, strides, kernel.ref_name,
                            groups))
            else:
                padding = [0] * 4
                padding[0] = total_pad_h / 2
                padding[1] = total_pad_h - padding[0]
                padding[2] = total_pad_w / 2
                padding[3] = total_pad_w - padding[2]
                code.append("{} = layers.pad2d({}, {})".format(
                    node.output_name, data.ref_name, padding))
                code.append(
                    "{} = layers.conv2d({}, {}, {}, stride={}, param_attr=\'{}\', bias_attr=False, groups={})"
                    .format(node.output_name, node.ref_name, in_channels,
                            [k_h, k_w], strides, kernel.ref_name, groups))
        else:
            code.append(
                "{} = layers.conv2d({}, {}, {}, stride={}, param_attr=\'{}\', bias_attr=False, groups={})"
                .format(node.output_name, data.ref_name, in_channels,
                        [k_h, k_w], strides, kernel.ref_name, groups))
        return code

    def emit_softmax(self, node):
        data = node.inputs[0]
        code = "{} = layers.softmax({})".format(node.output_name,
                                                data.ref_name)
        return code

