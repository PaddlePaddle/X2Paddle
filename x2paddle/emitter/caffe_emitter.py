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

import numbers
from x2paddle.parser.caffe_parser import CaffeGraph
from x2paddle.core.emitter import Emitter
from x2paddle.core.util import *


class CaffeEmitter(Emitter):
    def __init__(self, parser):
        super(CaffeEmitter, self).__init__()
        self.parser = parser
        self.graph = parser.caffe_graph
        self.weights = dict()
        self.resolver = parser.resolver

    def run(self):
        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                emit_func = getattr(self, op)
                emit_func(node)

        for i in range(len(self.graph.topo_sort)):
            node_name = self.graph.topo_sort[i]
            node = self.graph.get_node(node_name)
            for layer in node.fluid_code.layers:
                print(layer.get_code())

    @staticmethod
    def get_kernel_value(scalar, repeated, idx, default=None):
        if scalar:
            return scalar
        if repeated:
            if isinstance(repeated, numbers.Number):
                return repeated
            if len(repeated) == 1:
                # Same value applies to all spatial dimensions
                return int(repeated[0])
            assert idx < len(repeated)
            # Extract the value for the given spatial dimension
            return repeated[idx]
        if default is None:
            raise ValueError('Unable to determine kernel parameter!')
        return default

    def get_kernel_parameters(self, kind, params):
        assert kind in ['Convolution', 'Pooling', 'Deconvolution']

        k_h = self.get_kernel_value(params.kernel_h, params.kernel_size, 0)
        k_w = self.get_kernel_value(params.kernel_w, params.kernel_size, 1)
        s_h = self.get_kernel_value(params.stride_h,
                                    params.stride,
                                    0,
                                    default=1)
        s_w = self.get_kernel_value(params.stride_w,
                                    params.stride,
                                    1,
                                    default=1)
        p_h = self.get_kernel_value(params.pad_h, params.pad, 0, default=0)
        p_w = self.get_kernel_value(params.pad_w, params.pad, 1, default=0)
        dila_h = dila_w = 1
        group = 1
        c_o = 1
        if kind in ['Convolution', 'Deconvolution']:
            c_o = params.num_output
            group = params.group
            dila_len = len(params.dilation)
            if dila_len == 2:
                dila_h = params.dilation[0]
                dila_w = params.dilation[1]
            elif dila_len == 1:
                dila_h = dila_w = params.dilation[0]
            else:
                assert dila_len == 0, "invalid length[%s] of dilation in convolution" % (
                    dila_len)
        return c_o, k_h, k_w, s_h, s_w, p_h, p_w, dila_h, dila_w, group

    def Input(self, node):
        shape = list(node.layer.input_param.shape[0].dim)[1:]
        dtype = 'float32'
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("data",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)

    def Convolution(self, node):
        params = node.layer.convolution_param
        c_o, k_h, k_w, s_h, s_w, p_h, p_w, dila_h, dila_w, group = self.get_kernel_parameters(
            node.layer_type, params)
        assert len(node.inputs
                   ) == 1, 'The count of Convolution node\'s input is not 1.'
        input_name = node.inputs[0]
        input = self.graph.get_node_with_next(input_name,
                                              node,
                                              need=0,
                                              copy=True)
        attr = {
            'filter_size': [k_h, k_w],
            'num_filters': c_o,
            'stride': [s_h, s_w],
            'padding': [p_h, p_w],
            'dilation': [dila_h, dila_w],
            'groups': group,
            'name': string(node.layer_name),
            'param_attr': string(node.layer_name + '_weight'),
            'bias_attr': string(node.layer_name + '_bias'),
        }
        node.fluid_code.add_layer("conv2d",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Deconvolution(self, node):
        params = node.layer.convolution_param
        c_o, k_h, k_w, s_h, s_w, p_h, p_w, dila_h, dila_w, group = self.get_kernel_parameters(
            node.layer_type, params)
        assert len(node.inputs
                   ) == 1, 'The count of Deconvolution node\'s input is not 1.'
        input_name = node.inputs[0]
        input = self.graph.get_node_with_next(input_name,
                                              node,
                                              need=0,
                                              copy=True)
        attr = {
            'output_size': None,
            'filter_size': [k_h, k_w],
            'num_filters': c_o,
            'stride': [s_h, s_w],
            'padding': [p_h, p_w],
            'dilation': [dila_h, dila_w],
            'groups': group,
            'name': string(node.layer_name),
            'param_attr': string(node.layer_name + '_weight'),
            'bias_attr': string(node.layer_name + '_bias')
        }
        node.fluid_code.add_layer("conv2d_transpose",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Pooling(self, node):
        params = node.layer.pooling_param
        c_o, k_h, k_w, s_h, s_w, p_h, p_w, dila_h, dila_w, group = self.get_kernel_parameters(
            node.layer_type, params)
        if params.pool == 0:
            pool_type = 'max'
        else:
            pool_type = 'avg'
        assert len(
            node.inputs) == 1, 'The count of Pooling node\'s input is not 1.'
        input_name = node.inputs[0]
        input = self.graph.get_node_with_next(input_name,
                                              node,
                                              need=0,
                                              copy=True)
        attr = {
            'pool_size': [k_h, k_w],
            'num_filters': c_o,
            'pool_stride': [s_h, s_w],
            'pool_padding': [p_h, p_w],
            'ceil_mode': True,
            'pool_type': pool_type,
            'exclusive': True,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("pool2d",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def ReLU(self, node):
        assert len(
            node.inputs) == 1, 'The count of ReLU node\'s input is not 1.'
        input_name = node.inputs[0]
        input = self.graph.get_node_with_next(input_name,
                                              node,
                                              need=0,
                                              copy=True)
        attr = {'name': string(node.layer_name)}
        node.fluid_code.add_layer("relu",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def LRN(self, node):
        assert len(node.inputs) == 1, 'The count of LRN node\'s input is not 1.'
        params = node.layer.lrn_param
        # The window size must be an odd value. For a window
        # size of (2*n+1), Paddle defines depth_radius = n.
        assert params.local_size % 2 == 1
        # Caffe scales by (alpha/(2*n+1)), whereas Paddle
        # just scales by alpha (as does Krizhevsky's paper).
        # We'll account for that here.
        alpha = params.alpha / float(params.local_size)
        input_name = node.inputs[0]
        input = self.graph.get_node_with_next(input_name,
                                              node,
                                              need=0,
                                              copy=True)
        attr = {
            'n': params.local_size,
            'k': 1.0,
            'alpha': alpha,
            'beta': params.beta,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("lrn",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def InnerProduct(self, node):
        assert len(node.inputs
                   ) == 1, 'The count of InnerProduct node\'s input is not 1.'
        params = node.layer.inner_product_param
        assert params.axis == 1
        assert params.bias_term == True
        input_name = node.inputs[0]
        input = self.graph.get_node_with_next(input_name,
                                              node,
                                              need=0,
                                              copy=True)
        attr = {
            'size': params.num_output,
            'name': string(node.layer_name),
            'act': None,
            'param_attr': string(node.layer_name + '_weight'),
            'bias_attr': string(node.layer_name + '_bias')
        }
        node.fluid_code.add_layer("fc",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Softmax(self, node):
        assert len(
            node.inputs) == 1, 'The count of Softmax node\'s input is not 1.'
        input_name = node.inputs[0]
        input = self.graph.get_node_with_next(input_name,
                                              node,
                                              need=0,
                                              copy=True)
        params = node.layer.softmax_param
        axis = params.axis
        shape = node.input_shape[0]
        dims = len(shape)
        axis = axis + dims if axis < 0 else axis
        need_transpose = False
        if axis + 1 != dims:
            need_transpose = True
        if need_transpose:
            in_order = list(range(dims))
            in_order.remove(axis)
            in_order.append(axis)
            attr = {
                'perm': in_order,
                'name': string(node.layer_name + '_transpose_in')
            }
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
        attr = {'name': string(node.layer_name + '_softmax')}
        node.fluid_code.add_layer("softmax",
                                  inputs=input if need_transpose else node,
                                  output=node,
                                  param_attr=attr)
        if need_transpose:
            out_order = [
                0,
            ] * dims
            for id, v in enumerate(in_order):
                out_order[v] = id
            attr = {
                'perm': out_order,
                'name': string(node.layer_name + '_transpose_out')
            }
            node.fluid_code.add_layer("transpose",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)
