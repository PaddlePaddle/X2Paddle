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
from x2paddle.decoder.caffe_decoder import CaffeGraph
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.util import *


class CaffeOpMapper(OpMapper):
    def __init__(self, decoder):
        super(CaffeOpMapper, self).__init__()
        self.graph = decoder.caffe_graph
        self.weights = dict()
        resolver = decoder.resolver
        if resolver.has_pycaffe():
            self.did_use_pb = False
        else:
            self.did_use_pb = True

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
            node = self.graph.get_node(node_name)
            self.net_code += node.fluid_code.gen_codes()

    def adjust_parameters(self, node, data):
        if not self.did_use_pb:
            return data

        # When using the protobuf-backend, each parameter initially has four dimensions.
        # In certain cases (like FC layers), we want to eliminate the singleton dimensions.
        # This implementation takes care of the common cases. However, it does leave the
        # potential for future issues.
        # The Caffe-backend does not suffer from this problem.
        data = list(data)

        squeeze_indices = [1]  # Squeeze biases.
        if node.kind == NodeKind.InnerProduct:
            squeeze_indices.append(0)  # Squeeze FC.

        for idx in squeeze_indices:
            if idx >= len(data):
                continue

            d = data[idx]
            assert len(
                d.shape
            ) == 4, 'invalid shape[%s] from caffe when adjust_parameters' % (
                str(d.shape))

            shape_old = d.shape
            sq_axis = None
            if idx == 0:
                sq_axis = (0, 1)
            elif idx == 1:
                sq_axis = (0, 1, 2)
            else:
                continue

            data[idx] = np.squeeze(d, axis=sq_axis)
            shape_new = data[idx].shape
            if len(shape_old) != shape_new:
                debug('squeeze idx:%d, with kind:%s,name:%s' % \
                        (idx, node.kind, node.name))
        return data

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

        kernel = [k_h, k_w]
        stride = [s_h, s_w]
        pad = [p_h, p_w]
        dilation = [dila_h, dila_w]

        return c_o, kernel, stride, pad, dilation, group

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
        data = node.data
        data = self.adjust_parameters(node, data)
        self.weights[node.layer_name + '_weights'] = data[0]
        if len(data) == 2:
            self.weights[node.layer_name + '_bias'] = data[1]
        params = node.layer.convolution_param
        channel, kernel, stride, pad, dilation, group = self.get_kernel_parameters(
            node.layer_type, params)
        assert len(node.inputs
                   ) == 1, 'The count of Convolution node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        attr = {
            'filter_size': kernel,
            'num_filters': channel,
            'stride': stride,
            'padding': pad,
            'dilation': dilation,
            'groups': group,
            'name': string(node.layer_name),
            'param_attr': string(node.layer_name + '_weights'),
            'bias_attr': string(node.layer_name + '_bias'),
        }
        node.fluid_code.add_layer("conv2d",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Deconvolution(self, node):
        data = node.data
        data = self.adjust_parameters(node, data)
        self.weights[node.layer_name + '_weights'] = data[0]
        if len(data) == 2:
            self.weights[node.layer_name + '_bias'] = data[1]
        params = node.layer.convolution_param
        channel, kernel, stride, pad, dilation, group = self.get_kernel_parameters(
            node.layer_type, params)
        assert len(node.inputs
                   ) == 1, 'The count of Deconvolution node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        attr = {
            'output_size': None,
            'filter_size': kernel,
            'num_filters': channel,
            'stride': stride,
            'padding': pad,
            'dilation': dilation,
            'groups': group,
            'name': string(node.layer_name),
            'param_attr': string(node.layer_name + '_weights'),
            'bias_attr': string(node.layer_name + '_bias')
        }
        node.fluid_code.add_layer("conv2d_transpose",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Pooling(self, node):
        params = node.layer.pooling_param
        channel, kernel, stride, pad, dilation, group = self.get_kernel_parameters(
            node.layer_type, params)
        if params.pool == 0:
            pool_type = 'max'
        else:
            pool_type = 'avg'
        assert len(
            node.inputs) == 1, 'The count of Pooling node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        attr = {
            'pool_size': kernel,
            'pool_stride': stride,
            'pool_padding': pad,
            'ceil_mode': True,
            'pool_type': string(pool_type),
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
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
        data = node.data
        data = self.adjust_parameters(node, data)
        # Reshape the parameters to Paddle's ordering
        transpose_order = (1, 0)
        w = data[0]
        fc_shape = w.shape
        output_channels = fc_shape[0]
        w = w.reshape((output_channels, -1))
        w = w.transpose(transpose_order)
        data[0] = w

        self.weights[node.layer_name + '_weights'] = data[0]
        if len(data) == 2:
            self.weights[node.layer_name + '_bias'] = data[1]
        assert len(node.inputs
                   ) == 1, 'The count of InnerProduct node\'s input is not 1.'
        params = node.layer.inner_product_param
        assert params.axis == 1
        assert params.bias_term == True
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        attr = {
            'size': params.num_output,
            'name': string(node.layer_name),
            'act': None,
            'param_attr': string(node.layer_name + '_weights'),
            'bias_attr': string(node.layer_name + '_bias')
        }
        node.fluid_code.add_layer("fc",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Softmax(self, node):
        assert len(
            node.inputs) == 1, 'The count of Softmax node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
                                  inputs=node if need_transpose else input,
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
