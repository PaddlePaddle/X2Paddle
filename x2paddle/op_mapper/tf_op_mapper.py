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

from x2paddle.decoder.tf_decoder import TFGraph
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.util import *
import inspect
import numpy


# compute padding size for SAME mode
def get_same_padding(in_size, kernel_size, stride):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0
    return [pad0, pad1]


def nhwc_dim_to_nchw(node, dim):
    tf_data_format = list(node.tf_data_format)
    pd_data_format = list(node.pd_data_format)
    if isinstance(dim, list):
        for i in range(len(dim)):
            char = tf_data_format[dim[i]]
            dim[i] = pd_data_format.index(char)
    else:
        char = tf_data_format[dim]
        dim = pd_data_format.index(char)
    return dim

    if dim < 0:
        dim += 4
    if dim > 0:
        dim = (dim + 1) % 4 + int((dim + 1) / 4)
    return dim


class TFOpMapper(OpMapper):
    directly_map_ops = {
        'Relu': ['relu'],
        'Relu6': ['relu6'],
        'Shape': ['shape'],
        'Abs': ['abs'],
        'Sigmoid': ['sigmoid'],
        'Exp': ['exp'],
        'Rsqrt': ['rsqrt'],
        'swish_f32': ['swish']
    }
    elementwise_ops = {
        'Add': 'elementwise_add',
        'RealDiv': 'elementwise_div',
        'Sub': 'elementwise_sub',
        'Maximum': 'elementwise_max',
        'Mul': 'elementwise_mul'
    }

    def __init__(self, decoder):
        super(TFOpMapper, self).__init__()
        self.decoder = decoder
        self.graph = decoder.tf_graph
        self.weights = dict()
        self.omit_nodes = list()
        self.used_custom_layers = dict()

        not_placeholder = list()
        for name in self.graph.input_nodes:
            if self.graph.get_node(name).layer_type != "Placeholder":
                not_placeholder.append(name)
        for name in not_placeholder:
            idx = self.graph.input_nodes.index(name)
            del self.graph.input_nodes[idx]

        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if op in self.directly_map_ops:
                self.directly_map(node)
            elif op in self.elementwise_ops:
                self.elementwise_map(node)
            elif hasattr(self, op):
                func = getattr(self, op)
                func(node)
            else:
                raise Exception("OP: [{}] not support yet".format(op))

    def directly_map(self, node):
        assert node.layer_type in self.directly_map_ops
        op_info = self.directly_map_ops[node.layer_type]
        input = self.graph.get_node(node.layer.input[0], copy=True)
        attr = dict()
        for param in op_info[1:]:
            tf_param_name = list(param.keys())[0]
            pd_param_name = list(param.values())[0]
            tf_param = node.get_attr(tf_param_name)
            attr[pd_param_name] = tf_param
        node.fluid_code.add_layer(op_info[0],
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def elementwise_map(self, node):
        assert node.layer_type in self.elementwise_ops
        op_type = self.elementwise_ops[node.layer_type]
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        x_shape = x.out_shapes[0]
        y_shape = y.out_shapes[0]
        # incomplement broadcasting support for paddle
        x_input = x
        y_input = y
        if len(x_shape) < len(y_shape):
            unrevertable_ops = [
                "elementwise_sub", "elementwise_div", "elementwise_floordiv",
                "elementwise_mod", "elementwise_pow"
            ]
            if op_type not in unrevertable_ops:
                x_input = y
                y_input = x
                x_shape = y.out_shapes[0]
                y_shape = x.out_shapes[0]
            else:
                raise Exception("Unexpected situation happend")

        if len(x_shape) == 4 and len(y_shape) == 1:
            if x_input.tf_data_format == "NHWC":
                axis = 1
            else:
                axis = -1
            attr = {"axis": axis}
            inputs = {"x": x_input, "y": y_input}
            node.fluid_code.add_layer(op_type,
                                      inputs=inputs,
                                      output=node,
                                      param_attr=attr)
            return

        is_sub_seq = True
        for i in range(len(y_shape)):
            index = -1 * i - 1
            if y_shape[index] != x_shape[index]:
                is_sub_seq = False
        if not is_sub_seq:
            x_expand_times = [1] * len(x_shape)
            y_expand_times = [1] * len(y_shape)
            x_need_expand = False
            y_need_expand = False
            for i in range(len(y_shape)):
                index = -1 * i - 1
                if y_shape[index] != x_shape[index]:
                    if y_shape[index] == 1:
                        y_expand_times[index] = x_shape[index]
                        y_need_expand = True
                    elif x_shape[index] == 1:
                        x_expand_times[index] = y_shape[index]
                        x_need_expand = True
                    else:
                        raise Exception("Unexpected situation happend")
            if x_need_expand:
                if len(x_expand_times) == 3 and x.tf_data_format == "NHWC":
                    x_expand_times = [x_expand_times[i] for i in [2, 0, 1]]
                if len(x_expand_times) == 4 and x.tf_data_format == "NHWC":
                    x_expand_times = [x_expand_times[i] for i in [0, 3, 1, 2]]
                attr = {"expand_times": x_expand_times}
                node.fluid_code.add_layer("expand",
                                          inputs=x_input,
                                          output="x_tmp",
                                          param_attr=attr)
                x_input = "x_tmp"
            if y_need_expand:
                if len(y_expand_times) == 3 and y.tf_data_format == "NHWC":
                    y_expand_times = [y_expand_times[i] for i in [2, 0, 1]]
                if len(y_expand_times) == 4 and y.tf_data_format == "NHWC":
                    y_expand_times = [y_expand_times[i] for i in [0, 3, 1, 2]]
                attr = {"expand_times": y_expand_times}
                node.fluid_code.add_layer("expand",
                                          inputs=y_input,
                                          output="y_tmp",
                                          param_attr=attr)
                y_input = "y_tmp"
        inputs = {"x": x_input, "y": y_input}
        node.fluid_code.add_layer(op_type,
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Placeholder(self, node):
        shape = node.out_shapes[0]
        assert len(shape) != 0, "Unknown shape of input nodes[{}].".format(
            node.layer_name)
        if node.tf_data_format == "NHWC" and len(shape) == 4:
            shape = [shape[i] for i in [0, 3, 1, 2]]
        elif node.tf_data_format == "NCHW" and len(shape) == 4:
            self.graph.data_format_propagation(node)
        dtype = node.dtype
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name),
            'append_batch_size': False
        }
        node.fluid_code.add_layer("data",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)

    def Const(self, node):
        shape = node.out_shapes[0]
        dtype = node.dtype
        value = node.value
        initializer = "Constant(0.0)"
        if len(shape) == 0:
            assert value.size == 1, "Unexpected situation happend"
            shape = [1]
            initializer = "Constant({})".format(value)

        self.weights[node.layer_name] = node.value

        if node.tf_data_format == "NHWC":
            if len(shape) == 4:
                shape = [shape[i] for i in [0, 3, 1, 2]]
            if len(shape) == 3:
                shape = [shape[i] for i in [2, 0, 1]]
                self.weights[node.layer_name] = numpy.transpose(
                    node.value, (2, 0, 1))
        elif node.tf_data_format == "NCHW":
            if len(shape) == 4:
                self.graph.data_format_propagation(node)

        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name),
            'default_initializer': initializer
        }
        node.fluid_code.add_layer("create_parameter",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)

    def Transpose(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        perm = self.graph.get_node(node.layer.input[1], copy=True)
        assert perm.layer_type == "Const", "Perm of transpose OP should be Const"
        del self.weights[perm.layer_name.replace('/', '_')]
        perm.fluid_code.clear()
        perm = perm.value.tolist()

        if perm == [0, 3, 1, 2] and input.data_format == "NHWC":
            node.fluid_code.add_layer("assign",
                                      inputs=input,
                                      output=node,
                                      param_attr=None)
            node.tf_data_format = "NCHW"
            self.graph.data_format_propagation(node)
        elif perm == [0, 2, 3, 1] and input.tf_data_format == "NCHW":
            node.fluid_code.add_layer("assign",
                                      inputs=input,
                                      output=node,
                                      param_attr=None)
            node.tf_data_format = "NHWC"
            self.graph.data_format_propagation(node)
        elif len(input.out_shapes[0]) > 4:
            print(input.layer_name, input.tf_data_format, input.pd_data_format)
            tf_data_format = list(input.tf_data_format)
            pd_data_format = list(input.pd_data_format)
            new_perm = [i for i in range(len(perm))]
            for i in range(len(perm)):
                char0 = tf_data_format[i]
                char1 = tf_data_format[perm[i]]
                index0 = pd_data_format.index(char0)
                index1 = pd_data_format.index(char1)
                new_perm[index0] = index1
            node.tf_data_format = [tf_data_format[i] for i in perm]
            node.pd_data_format = [pd_data_format[i] for i in perm]
            attr = {'perm': new_perm}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
        elif len(node.out_shapes[0]) != 4:
            attr = {'perm': perm}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
        else:
            raise Exception("Unexpected situation happend in Transpose OP")

    def MaxPool(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)

        in_shape = input.out_shapes[0]
        if in_shape.count(-1) > 2:
            in_shape = self.decoder.infer_tensor(input).shape

        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"
        padding = 0

        if not channel_first:
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            k_size = [k_size[i] for i in [0, 3, 1, 2]]
        else:
            self.graph.data_format_propagation(node)

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[2], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[3], strides[3])
            pad_h = pad_h[0] + pad_h[1]
            pad_w = pad_w[0] + pad_w[1]
            attr = {"paddings": [0, pad_h, 0, pad_w], "pad_value": -10000.0}
            node.fluid_code.add_layer("pad2d",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            input = node
        attr = {
            "pool_size": k_size[2:4],
            "pool_type": string("max"),
            "pool_padding": padding,
            "pool_stride": strides[2:4]
        }
        node.fluid_code.add_layer("pool2d",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Conv2D(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        assert kernel.layer_type == "Const", "Kernel of Conv2D should be Const"
        self.omit_nodes.append(kernel.layer_name)

        node.fluid_code.add_note("#{} : {}".format(node.layer.name,
                                                   node.layer_name))

        in_shape = input.out_shapes[0]
        if in_shape.count(-1) > 2:
            in_shape = self.decoder.infer_tensor(input).shape
        k_size = kernel.out_shapes[0]
        if k_size.count(-1) > 2:
            k_size = self.decoder.infer_tensor(kernel).shape

        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"
        padding = 0

        self.weights[kernel.layer_name.replace('/', '_')] = numpy.transpose(
            kernel.value, (3, 2, 0, 1))

        if not channel_first:
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
        else:
            self.graph.data_format_propagation(node)

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            if pad_h[0] == pad_h[1] and pad_w[0] == pad_w[1]:
                padding = [pad_h[0], pad_w[0]]
            else:
                attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}
                node.fluid_code.add_layer("pad2d",
                                          inputs=input,
                                          output=node,
                                          param_attr=attr)
                input = node
        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": k_size[3],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4],
            "padding": padding
        }
        node.fluid_code.add_layer("conv2d",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def BiasAdd(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        bias = self.graph.get_node(node.layer.input[1], copy=True)
        axis = -1
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            axis = 1
        inputs = {"x": input, "y": bias}
        attr = {"axis": axis}
        node.fluid_code.add_layer("elementwise_add",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def FusedBatchNorm(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        gamma = self.graph.get_node(node.layer.input[1], copy=True)
        beta = self.graph.get_node(node.layer.input[2], copy=True)
        moving_mean = self.graph.get_node(node.layer.input[3], copy=True)
        moving_var = self.graph.get_node(node.layer.input[4], copy=True)
        data_format = node.get_attr("data_format").decode()
        channel_first = data_format == "NCHW"

        assert gamma.layer_type == "Const"
        assert beta.layer_type == "Const"
        assert moving_mean.layer_type == "Const"
        assert moving_var.layer_type == "Const"
        self.omit_nodes.append(gamma.layer_name)
        self.omit_nodes.append(beta.layer_name)
        self.omit_nodes.append(moving_mean.layer_name)
        self.omit_nodes.append(moving_var.layer_name)

        if channel_first:
            self.data_format_propagation(node)

        attr = {
            "epsilon": node.get_attr("epsilon"),
            "param_attr": string(gamma.layer_name),
            "bias_attr": string(beta.layer_name),
            "moving_mean_name": string(moving_mean.layer_name),
            "moving_variance_name": string(moving_var.layer_name),
            "is_test": True
        }

        node.fluid_code.add_layer("batch_norm",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def DepthwiseConv2dNative(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        assert kernel.layer_type == "Const", "Kernel of DepthwiseConv2DNative should be Const"
        self.omit_nodes.append(kernel.layer_name)

        node.fluid_code.add_note("#{} : {}".format(node.layer.name,
                                                   node.layer_name))

        in_shape = input.out_shapes[0]
        if in_shape.count(-1) > 2:
            in_shape = self.decoder.infer_tensor(input).shape
        k_size = kernel.out_shapes[0]
        if k_size.count(-1) > 2:
            k_size = self.decoder.infer_tensor(kernel).shape

        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"
        padding = 0

        self.weights[kernel.layer_name.replace('/', '_')] = numpy.transpose(
            kernel.value, (2, 3, 0, 1))

        if not channel_first:
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
        else:
            self.data_format_propagation(node)

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            if pad_h[0] == pad_h[1] and pad_w[0] == pad_w[1]:
                padding = [pad_h[0], pad_w[0]]
            else:
                attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}
                node.fluid_code.add_layer("pad2d",
                                          inputs=input,
                                          output=node,
                                          param_attr=attr)
                input = node

        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": in_shape[1],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4],
            "groups": k_size[3] * in_shape[1],
            "padding": padding
        }
        node.fluid_code.add_layer("conv2d",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Reshape(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        param = self.graph.get_node(node.layer.input[1], copy=True)
        if param.layer_type == "Const":
            attr = {"shape": param.value.tolist()}
            self.omit_nodes.append(param.layer_name)
        else:
            # Here is a trick method to solove tensor parameter in tensorflow
            shape = self.decoder.infer_shape_tensor(param, node.out_shapes[0])
            if shape.count(-1) <= 1:
                attr = {"shape": shape}
                self.omit_nodes.append(param.layer_name)
            else:
                assert len(param.out_shapes[0]
                           ) == 1, "Unexpected situation of shape parameter"
                attr = {"shape": [-1]}
                node.fluid_code.add_layer("reshape",
                                          inputs=param,
                                          output="shape_param",
                                          param_attr=attr)
                attr = {"num_or_sections": param.out_shapes[0][0], "dim": 0}
                node.fluid_code.add_layer("split",
                                          inputs="shape_param",
                                          output=node,
                                          param_attr=attr)
                new_param = "["
                for i in range(param.out_shapes[0][0]):
                    new_param += (node.layer_name + "[{}]".format(i) + ", ")
                new_param = new_param.strip(", ") + "]"
                attr = {"shape": new_param}
        if len(attr["shape"]) == 4 and node.tf_data_format == "NHWC":
            attr["shape"] = [attr["shape"][i] for i in [0, 3, 1, 2]]
        node.fluid_code.add_layer("reshape",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def AvgPool(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)

        in_shape = input.out_shapes[0]
        if in_shape.count(-1) > 2:
            in_shape = self.decoder.infer_tensor(input).shape

        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            k_size = [k_size[i] for i in [0, 3, 1, 2]]
        else:
            self.graph.data_format_propagation(node)

        attr = {
            "pool_size": k_size[2:4],
            "pool_type": string("avg"),
            "pool_stride": strides[2:4]
        }
        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[2], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[3], strides[3])
            assert pad_h[0] == pad_h[1] and pad_w[0] == pad_w[
                1], "Cannot map AvgPool"
            attr["pool_padding"] = [pad_h[0], pad_w[0]]
        node.fluid_code.add_layer("pool2d",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def SplitV(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        num_sections = self.graph.get_node(node.layer.input[1], copy=True)
        dim = self.graph.get_node(node.layer.input[2], copy=True)
        assert num_sections.layer_type == "Const"
        assert dim.layer_type == "Const"
        self.omit_nodes.append(num_sections.layer_name)
        self.omit_nodes.append(dim.layer_name)
        dim = dim.value
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            dim = nhwc_dim_to_nchw(input, dim)
        attr = {
            "num_or_sections": num_sections.value.tolist(),
            "dim": dim.value
        }
        node.fluid_code.add_layer("split",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def ConcatV2(self, node):
        inputs = [
            self.graph.get_node(name, copy=True)
            for name in node.layer.input[:-1]
        ]
        axis = self.graph.get_node(node.layer.input[-1], copy=True)
        assert axis.layer_type == "Const"
        self.omit_nodes.append(axis.layer_name)
        axis = axis.value
        if inputs[0].tf_data_format == "NHWC" and len(
                inputs[0].out_shapes[0]) == 4:
            axis = nhwc_dim_to_nchw(inputs[0], axis)
        attr = {"axis": axis}
        node.fluid_code.add_layer("concat",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def Tile(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        expand_times = self.graph.get_node(node.layer.input[1], copy=True)
        assert expand_times.layer_type == "Const"
        self.omit_nodes.append(expand_times.layer_name)
        expand_times = expand_times.value.tolist()
        if input.tf_data_format == "NHWC":
            if len(input.out_shapes[0]) == 4:
                expand_times = [expand_times[i] for i in [0, 3, 1, 2]]
            elif len(input.out_shape[0]) == 3:
                expand_times = [expand_times[i] for i in [2, 0, 1]]
        attr = {"expand_times": expand_times}
        node.fluid_code.add_layer("expand",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Pack(self, node):
        inputs = [
            self.graph.get_node(name, copy=True) for name in node.layer.input
        ]
        axis = node.get_attr("axis")
        if inputs[0].tf_data_format == "NHWC" and len(
                inputs[0].out_shapes[0]) == 4:
            tf_data_format = list(inputs[0].tf_data_format)
            tf_data_format.insert(axis, str(len(tf_data_format)))
            axis = nhwc_dim_to_nchw(inputs[0], axis)
            pd_data_format = list(inputs[0].pd_data_format)
            pd_data_format.insert(axis, str(len(pd_data_format)))
            node.tf_data_format = "".join(tf_data_format)
            node.pd_data_format = "".join(pd_data_format)

        attr = {"axis": axis}
        node.fluid_code.add_layer("stack",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def Pad(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        paddings = self.graph.get_node(node.layer.input[1], copy=True)
        assert paddings.layer_type == "Const", "Padding should be Const"
        self.omit_nodes.append(paddings.layer_name)
        paddings = paddings.value.flatten().tolist()
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            paddings = [paddings[i] for i in [0, 1, 6, 7, 2, 3, 4, 5]]
        attr = {"paddings": paddings}
        node.fluid_code.add_layer("pad",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Range(self, node):
        start = self.graph.get_node(node.layer.input[0], copy=True)
        limit = self.graph.get_node(node.layer.input[1], copy=True)
        delta = self.graph.get_node(node.layer.input[2], copy=True)
        if start.layer_type == "Const":
            self.omit_nodes.append(start.layer_name)
            start = start.value
        if limit.layer_type == "Const":
            self.omit_nodes.append(limit.layer_name)
            limit = limit.value
        if delta.layer_type == "Const":
            self.omit_nodes.append(delta.layer_name)
            delta = delta.value
        inputs = {"start": start, "end": limit, "step": delta}
        attr = {"dtype": string(node.dtype)}
        node.fluid_code.append("range",
                               inputs=inputs,
                               output=node,
                               param_attr=None)

    def Mean(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        dims = reduce_idx.value.tolist()
        keep_dims = node.get_attr("keep_dims")

        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            for i in range(len(dims)):
                dims[i] = nhwc_dim_to_nchw(input, dims[i])

        attr = {"dim": dims, "keep_dim": keep_dims}
        node.fluid_code.add_layer("reduce_mean",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def MatMul(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        transpose_a = node.get_attr('transpose_a')
        transpose_b = node.get_attr('transpose_b')
        inputs = {"x": x, "y": y}
        # fix paddle shape infer problem
        # should be removed after paddle 1.6
        if x.out_shapes[0][-1] < 0 and y.out_shapes[0][0] > 0:
            shape = x.out_shapes[0]
            shape[-1] = y.out_shapes[0][0]
            attr = {"shape": shape}
            node.fluid_code.add_layer("reshape",
                                      inputs=x,
                                      output=x,
                                      param_attr=attr)
        attr = {"transpose_x": transpose_a, "transpose_y": transpose_b}
        node.fluid_code.add_layer("matmul",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def ArgMax(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        axis = self.graph.get_node(node.layer.input[1], copy=True)
        assert axis.layer_type == "Const", "ArgMax only support Const parameter"
        self.omit_nodes.append(axis.layer_name)
        axis = axis.value
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            axis = nhwc_dim_to_nchw(input, axis)
        attr = {"axis": axis}
        node.fluid_code.add_layer("argmax",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def StridedSlice(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        begin = self.graph.get_node(node.layer.input[1], copy=True)
        end = self.graph.get_node(node.layer.input[2], copy=True)
        strides = self.graph.get_node(node.layer.input[3], copy=True)
        assert begin.layer_type == "Const"
        assert end.layer_type == "Const"
        assert strides.layer_type == "Const"
        self.omit_nodes.append(begin.layer_name)
        self.omit_nodes.append(end.layer_name)
        self.omit_nodes.append(strides.layer_name)
        strides = strides.value.tolist()
        assert len(set(strides)) == 1 and strides[0] == 1

        begin = begin.value.tolist()
        end = end.value.tolist()
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            begin = [begin[i] for i in [0, 3, 1, 2]]
            end = [end[i] for i in [0, 3, 1, 2]]

        attr = {"axes": range(len(strides)), "starts": begin, "ends": end}
        node.fluid_code.add_layer("slice",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Slice(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        begin = self.graph.get_node(node.layer.input[1], copy=True)
        size = self.graph.get_node(node.layer.input[2], copy=True)
        #        assert begin.layer_type == "Const"
        #        assert size.layer_type == "Const"
        self.omit_nodes.append(begin.layer_name)
        self.omit_nodes.append(size.layer_name)
        if begin.layer_type == "Const":
            begin = begin.value.tolist()
        else:
            begin = self.decoder.infer_tensor(begin).tolist()
        if size.layer_type == "const":
            size = size.value.tolist()
        else:
            size = self.decoder.infer_tensor(size).tolist()

        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            size = [size[i] for i in [0, 3, 1, 2]]
            begin = [begin[i] for i in [0, 3, 1, 2]]

        attr = {"shape": size, "offsets": begin}
        node.fluid_code.add_layer("crop",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Conv2DBackpropInput(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        assert kernel.layer_type == "Const", "Kernel of Conv2DBackpropInput should be Const"
        self.omit_nodes.append(kernel.layer_name)

        node.fluid_code.add_note("#{} : {}".format(node.layer.name,
                                                   node.layer_name))

        in_shape = input.out_shapes[0]
        if in_shape.count(-1) > 2:
            in_shape = self.decoder.infer_tensor(input).shape
        k_size = kernel.out_shapes[0]
        if k_size.count(-1) > 2:
            k_size = self.decoder.infer_tensor(kernel).shape

        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"
        self.weights[kernel.layer_name.replace('/', '_')] = numpy.transpose(
            kernel.value, (3, 2, 0, 1))

        if not channel_first:
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
        else:
            self.data_format_propagation(node)

        padding = 0
        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            if pad_h[0] == pad_h[1] and pad_w[0] == pad_w[1]:
                padding = [pad_h[0], pad_w[0]]
            else:
                attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}
                node.fluid_code.add_layer("pad2d",
                                          inputs=input,
                                          output=node,
                                          param_attr=attr)
                input = node
        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": k_size[3],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4],
            "padding": padding
        }
        node.fluid_code.add_layer(
            "conv2d_transpose",
            inputs=input if channel_first and pad_mode != "SAME" else node,
            output=node,
            param_attr=attr)

    def Max(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        dim = reduce_idx.value.tolist()
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            dim = nhwc_dim_to_nchw(input, dim)

        attr = {"dim": dim, "keep_dim": keep_dims}
        node.fluid_code.add_layer("reduce_max",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Sum(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        dim = reduce_idx.value.tolist()
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            dim = nhwc_dim_to_nchw(input, dim)

        attr = {"dim": dim, "keep_dim": keep_dims}
        node.fluid_code.add_layer("reduce_sum",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Cast(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        dtype = node.dtype_map[node.get_attr('DstT')]
        attr = {"dtype": string(dtype)}
        node.fluid_code.add_layer("cast",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def FloorDiv(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {'x': x, 'y': y}
        node.fluid_code.add_layer("elementwise_div",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)
        node.fluid_code.add_layer("floor",
                                  inputs=node,
                                  output=node,
                                  param_attr=None)

    def Split(self, node):
        dim = self.graph.get_node(node.layer.input[0], copy=True)
        input = self.graph.get_node(node.layer.input[1], copy=True)
        assert dim.layer_type == "Const"
        self.omit_nodes.append(dim.layer_name)
        num_split = node.get_attr('num_split')
        dim = dim.value
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            dim = nhwc_dim_to_nchw(input, dim)

        attr = {"num_or_sections": num_split, "dim": dim}
        node.fluid_code.add_layer("split",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Squeeze(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        squeeze_dims = node.get_attr('squeeze_dims')
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            for i in range(len(squeeze_dims)):
                squeeze_dims[i] = nhwc_dim_to_nchw(input, squeeze_dims[i])
        attr = {"axes": squeeze_dims}
        node.fluid_code.add_layer("squeeze",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Softmax(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        axis = node.get_attr("axis")
        if input.tf_data_format == "NHWC" and len(input.out_shapes[0]) == 4:
            axis = nhwc_dim_to_nchw(input, axis)
        attr = {"axis": axis}
        node.fluid_code.add_layer("softmax",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)
