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
import numpy


class TFOpMapper(OpMapper):
    def __init__(self, decoder):
        super(TFOpMapper, self).__init__()
        self.graph = decoder.tf_graph
        self.weights = dict()
        self.omit_nodes = list()

    def run(self):
        print("Total nodes: {}".format(len(self.graph.topo_sort)))

        # check if ops in model are all supported
        if not self.op_checker():
            raise Exception("Model are not supported yet.")

        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                func = getattr(self, op)
                func(node)

        for i in range(len(self.graph.topo_sort)):
            node_name = self.graph.topo_sort[i]
            if node_name in self.omit_nodes:
                continue
            node = self.graph.get_node(node_name)
            self.net_code += node.fluid_code.gen_codes()

    def Placeholder(self, node):
        shape = node.out_shapes[0]
        assert len(shape) != 0, "Unknown shape of input nodes[{}].".format(
            node.layer_name)
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
        self.weights[node.layer_name.replace('/', '_')] = node.value

    def Transpose(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        perm = self.graph.get_node(node.layer.input[1], copy=True)
        assert perm.layer_type == "Const", "Perm of transpose OP should be Const"
        del self.weights[perm.layer_name.replace('/', '_')]
        perm.fluid_code.clear()
        perm = perm.value.tolist()

        attr = {'perm': perm}
        node.fluid_code.add_layer("transpose",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def RealDiv(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {'x': x, 'y': y}
        node.fluid_code.add_layer("elementwise_div",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Relu(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("relu",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def Squeeze(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        squeeze_dims = node.get_attr('squeeze_dims')
        attr = {'axes': squeeze_dims}
        node.fluid_code.add_layer("squeeze",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def BiasAdd(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        bias = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {'x': input, 'y': bias}
        node.fluid_code.add_layer("elementwise_add",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Identity(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("assign",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def MaxPool(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        in_shape = input.out_shapes[0]
        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            pad_h = pad_h[0] + pad_h[1]
            pad_w = pad_w[0] + pad_w[1]
            attr = {"paddings": [0, pad_h, 0, pad_w], "pad_value": -10000.0}
            if pad_h + pad_w != 0:
                node.fluid_code.add_layer(
                    "pad2d",
                    inputs=input if channel_first else node,
                    output=node,
                    param_attr=attr)
        attr = {
            "pool_size": k_size[1:3],
            "pool_type": string("max"),
            "pool_stride": strides[2:4]
        }
        node.fluid_code.add_layer(
            "pool2d",
            inputs=input if channel_first and pad_mode != "SAME" else node,
            output=node,
            param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer("transpose",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)

    def Conv2D(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        assert kernel.layer_type == "Const", "Kernel of Conv2D should be Const"
        self.omit_nodes.append(kernel.layer_name)

        in_shape = input.out_shapes[0]
        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            self.weights[kernel.layer_name.replace('/', '_')] = numpy.transpose(
                kernel.value, (3, 2, 0, 1))
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}
            if pad_h[0] + pad_h[1] + pad_w[0] + pad_w[1] != 0:
                node.fluid_code.add_layer(
                    "pad2d",
                    inputs=input if channel_first else node,
                    output=node,
                    param_attr=attr)
        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": k_size[3],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4]
        }
        node.fluid_code.add_layer(
            "conv2d",
            inputs=input if channel_first and pad_mode != "SAME" else node,
            output=node,
            param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer("transpose",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)

    def Relu6(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("relu6",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def FusedBatchNorm(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        gamma = self.graph.get_node(node.layer.input[1], copy=True)
        beta = self.graph.get_node(node.layer.input[2], copy=True)
        moving_mean = self.graph.get_node(node.layer.input[3], copy=True)
        moving_var = self.graph.get_node(node.layer.input[4], copy=True)

        assert gamma.layer_type == "Const"
        assert beta.layer_type == "Const"
        assert moving_mean.layer_type == "Const"
        assert moving_var.layer_type == "Const"
        self.omit_nodes.append(gamma.layer_name)
        self.omit_nodes.append(beta.layer_name)
        self.omit_nodes.append(moving_mean.layer_name)
        self.omit_nodes.append(moving_var.layer_name)

        attr = {
            "epsilon": node.get_attr("epsilon"),
            "param_attr": string(gamma.layer_name),
            "data_layout": string(node.get_attr("data_format").decode()),
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

        in_shape = input.out_shapes[0]
        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            self.weights[kernel.layer_name.replace('/', '_')] = numpy.transpose(
                kernel.value, (2, 3, 0, 1))
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}
            if pad_h[0] + pad_h[1] + pad_w[0] + pad_w[1] != 0:
                node.fluid_code.add_layer("pad2d",
                                          inputs=input if channel_first
                                          and pad_mode != "SAME" else node,
                                          output=node,
                                          param_attr=attr)
        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": in_shape[1],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4],
            "groups": k_size[3] * in_shape[1]
        }
        node.fluid_code.add_layer("conv2d",
                                  inputs=input if channel_first else node,
                                  output=node,
                                  param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer("transpose",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)

    def Shape(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("shape",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def Reshape(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        param = self.graph.get_node(node.layer.input[1], copy=True)
        if param.layer_type == "Const":
            attr = {"shape": param.value.tolist()}
            self.omit_nodes.append(param.layer_name)
        else:
            # Here is a trick method to solove tensor parameter in tensorflow
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
        node.fluid_code.add_layer("reshape",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Add(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": x, "y": y}
        node.fluid_code.add_layer("elementwise_add",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def AvgPool(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        in_shape = input.out_shapes[0]
        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]

        attr = {
            "pool_size": k_size[1:3],
            "pool_type": string("avg"),
            "pool_stride": strides[2:4]
        }
        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            assert pad_h[0] == pad_h[1] and pad_w[0] == pad_w[
                1], "Cannot map AvgPool"
            attr["pool_padding"] = [pad_h[0], pad_w[0]]
        node.fluid_code.add_layer("pool2d",
                                  inputs=input if channel_first else node,
                                  output=node,
                                  param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer("transpose",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)

    def Softmax(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("softmax",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def Sigmoid(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("sigmoid",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def Maximum(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": x, "y": y}
        node.fluid_code.add_layer("elementwise_max",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def SplitV(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        num_sections = self.graph.get_node(node.layer.input[1], copy=True)
        dim = self.graph.get_node(node.layer.input[2], copy=True)
        assert num_sections.layer_type == "Const"
        assert dim.layer_type == "Const"
        self.omit_nodes.append(num_sections.layer_name)
        self.omit_nodes.append(dim.layer_name)
        attr = {
            "num_or_sections": num_sections.value.tolist(),
            "dim": dim.value
        }
        node.fluid_code.add_layer("split",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Exp(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("exp",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def ConcatV2(self, node):
        inputs = [
            self.graph.get_node(name, copy=True)
            for name in node.layer.input[:-1]
        ]
        axis = self.graph.get_node(node.layer.input[-1], copy=True)
        assert axis.layer_type == "Const"
        self.omit_nodes.append(axis.layer_name)
        attr = {"axis": axis.value}
        node.fluid_code.add_layer("concat",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def Tile(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        expand_times = self.graph.get_node(node.layer.input[1], copy=True)
        assert expand_times.layer_type == "Const"
        self.omit_nodes.append(expand_times.layer_name)
        attr = {"expand_times": expand_times.value.tolist()}
        node.fluid_code.add_layer("expand",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Pack(self, node):
        inputs = [
            self.graph.get_node(name, copy=True) for name in node.layer.input
        ]
        node.fluid_code.add_layer("stack",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Pad(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        paddings = self.graph.get_node(Node.layer.input[1], copy=True)
        assert paddings.layer_type == "Const", "Padding should be Const"
        self.omit_nodes.append(paddings.layer_name)
        attr = {"paddings": paddings.value.tolist()}
        node.fluid_code.add_layer("pad",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

#    def ResizeNearestNeighbor(self, node):
#        pass

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


#    def Fill(self, node):
#        shape = self.graph.get_node(node.layer

    def Mul(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": x, "y": y}
        node.fluid_code.add_layer("elementwise_mul",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Sub(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": x, "y": y}
        node.fluid_code.add_layer("elementwise_sub",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Rsqrt(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("rsqrt",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def swish_f32(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("sigmoid",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)
        inputs = {"x": input, "y": node}
        node.fluid_code.add_layer("elementwise_mul",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Mean(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        attr = {"dim": reduce_idx.value.tolist(), "keep_dim": keep_dims}
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
        attr = {"axis": axis.value}
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

        attr = {
            "axes": range(len(strides)),
            "starts": begin.value.tolist(),
            "ends": end.value.tolist()
        }
        node.fluid_code.add_layer("slice",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Slice(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        begin = self.graph.get_node(node.layer.input[1], copy=True)
        size = self.graph.get_node(node.layer.input[2], copy=True)
        assert begin.layer_type == "Const"
        assert size.layer_type == "Const"
        self.omit_nodes.append(begin.layer_name)
        self.omit_nodes.append(size.layer_name)

        attr = {"shape": size.value.tolist(), "offsets": begin.value.tolist()}
        node.code.add_layer("crop", inputs=input, output=node, param_attr=attr)

    def Abs(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("abs",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def Conv2DBackpropInput(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        assert kernel.layer_type == "Const", "Kernel of Conv2DBackpropInput should be Const"
        self.omit_nodes.append(kernel.layer_name)

        in_shape = input.out_shapes[0]
        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            self.weights[kernel.layer_name.replace('/', '_')] = numpy.transpose(
                kernel.value, (3, 2, 0, 1))
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}
            if pad_h[0] + pad_h[1] + pad_w[0] + pad_w[1] != 0:
                node.fluid_code.add_layer(
                    "pad2d",
                    inputs=input if channel_first else node,
                    output=node,
                    param_attr=attr)
        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": k_size[3],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4]
        }
        node.fluid_code.add_layer(
            "conv2d_transpose",
            inputs=input if channel_first and pad_mode != "SAME" else node,
            output=node,
            param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer("transpose",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)

    def Max(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        attr = {"dim": reduce_idx.value.tolist(), "keep_dim": keep_dims}
        node.fluid_code.add_layer("reduce_max",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Sum(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        attr = {"dim": reduce_idx.value.tolist(), "keep_dim": keep_dims}
        node.fluid_code.add_layer("reduce_sum",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def GreaterEqual(self, node):
        pass

    def RandomUniform(self, node):
        pass

    def cast(self, node):
        pass

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
