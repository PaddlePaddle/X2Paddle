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
import numpy as np
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

        k_h = self.get_kernel_value(params.kernel_h,
                                    params.kernel_size,
                                    0,
                                    default=1)
        k_w = self.get_kernel_value(params.kernel_w,
                                    params.kernel_size,
                                    1,
                                    default=1)
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

    def get_input_name(self, node):
        if hasattr(node, "index"):
            return node.layer_name + "[{}]".format(node.index)
        else:
            return node.layer_name

    def is_BN(self, node):
        return True if node.layer_type == 'BatchNorm' else False

    def is_Scale(self, node):
        return True if node.layer_type == 'Scale' else False

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
        assert data is not None, 'The parameter of {} (type is {}) is not set. You need to use python package of caffe to set the default value.'.format(
            node.layer_name, node.layer_type)
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
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp

        attr = {
            'filter_size':
            kernel,
            'num_filters':
            channel,
            'stride':
            stride,
            'padding':
            pad,
            'dilation':
            dilation,
            'groups':
            group,
            'name':
            string(node.layer_name),
            'param_attr':
            string(node.layer_name + '_weights'),
            'bias_attr':
            False if len(data) == 1 else string(node.layer_name + '_bias'),
        }
        node.fluid_code.add_layer("conv2d",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Deconvolution(self, node):
        data = node.data
        assert data is not None, 'The parameter of {} (type is {}) is not set. You need to use python package of caffe to set the default value.'.format(
            node.layer_name, node.layer_type)
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
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        attr = {
            'output_size':
            None,
            'filter_size':
            kernel,
            'num_filters':
            channel,
            'stride':
            stride,
            'padding':
            pad,
            'dilation':
            dilation,
            'groups':
            group,
            'name':
            string(node.layer_name),
            'param_attr':
            string(node.layer_name + '_weights'),
            'bias_attr':
            False if len(data) == 1 else string(node.layer_name + '_bias')
        }
        node.fluid_code.add_layer("conv2d_transpose",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Pooling(self, node):
        params = node.layer.pooling_param
        ceil_mode = getattr(params, 'ceil_mode', True)
        global_pool = getattr(params, 'global_pooling', False)
        kernel_default = [1, 1]
        channel, kernel, stride, pad, dilation, group = self.get_kernel_parameters(
            node.layer_type, params)
        if params.pool == 0:
            pool_type = 'max'
        else:
            pool_type = 'avg'
        assert len(
            node.inputs) == 1, 'The count of Pooling node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        attr = {
            'pool_size': kernel,
            'pool_stride': stride,
            'pool_padding': pad,
            'ceil_mode': ceil_mode,
            'pool_type': string(pool_type),
            'exclusive': True,
            'global_pooling': global_pool,
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
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
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
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
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
        assert data is not None, 'The parameter of {} (type is {}) is not set. You need to use python package of caffe to set the default value.'.format(
            node.layer_name, node.layer_type)
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
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        attr = {
            'size':
            params.num_output,
            'name':
            string(node.layer_name),
            'act':
            None,
            'param_attr':
            string(node.layer_name + '_weights'),
            'bias_attr':
            False if len(data) == 1 else string(node.layer_name + '_bias')
        }
        node.fluid_code.add_layer("fc",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Softmax(self, node):
        assert len(
            node.inputs) == 1, 'The count of Softmax node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
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

    def Slice(self, node):
        assert len(
            node.inputs) == 1, 'The count of Slice node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        params = node.layer.slice_param
        axis = params.axis
        points = list(params.slice_point)
        maxint32 = 2147483647
        points = [0] + points
        points.append(maxint32)
        i = 0
        node.fluid_code.add_note('{} = []'.format(node.layer_name))
        for i in range(len(points)):
            attr = {
                'axes': [axis],
                'starts': [points[i]],
                'ends': [points[i + 1]],
                'name': string(node.layer_name + '_' + str(i))
            }
            node.fluid_code.add_layer("slice",
                                      inputs=input,
                                      output=string(node.layer_name + '_' +
                                                    str(i)),
                                      param_attr=attr)
            node.fluid_code.add_note('{}.append({})'.format(
                node.layer_name, node.layer_name + '_' + str(i)))
            if i == len(points) - 2:
                break

    def Concat(self, node):
        assert len(
            node.inputs
        ) > 1, 'The count of Concat node\'s input is not more than 1.'
        inputs = []
        for i in range(len(node.inputs)):
            input = self.graph.get_bottom_node(node, idx=i, copy=True)
            if self.is_Scale(input):
                tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
                if self.is_BN(tmp):
                    input = tmp
            inputs.append(input)
        params = node.layer.concat_param
        axis = params.axis
        attr = {'axis': axis, 'name': string(node.layer_name)}
        node.fluid_code.add_layer("concat",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def PReLU(self, node):
        assert len(
            node.inputs) == 1, 'The count of PReLU node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        params = node.layer.prelu_param
        mode_bool = params.channel_shared
        if mode_bool:
            mode = 'all'
        else:
            mode = 'channel'
        data = node.data
        assert data is not None, 'The parameter of {} (type is {}) is not set. You need to use python package of caffe to set the default value.'.format(
            node.layer_name, node.layer_type)
        self.weights[node.layer_name + '_weights'] = data[0]
        attr = {
            'mode': mode,
            'param_attr': string(node.layer_name + '_weights'),
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("prelu",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Sigmoid(self, node):
        assert len(
            node.inputs) == 1, 'The count of PReLU node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        attr = {'name': string(node.layer_name)}
        node.fluid_code.add_layer("sigmoid",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def AbsVal(self, node):
        assert len(
            node.inputs) == 1, 'The count of PReLU node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        attr = {'name': string(node.layer_name)}
        node.fluid_code.add_layer("absval",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Accuracy(self, node):
        assert len(
            node.inputs) == 2, 'The count of Accuracy node\'s input is not 2.'
        inputs = []
        inputs[0] = None
        inputs[1] = None
        i = 0
        for shape in node.input_shape:
            if shape[1] == 1:
                input = self.graph.get_bottom_node(node, idx=i, copy=True)
                if self.is_Scale(input):
                    tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
                    if self.is_BN(tmp):
                        input = tmp
                inputs[1] = input
            else:
                input = self.graph.get_bottom_node(node, idx=i, copy=True)
                if self.is_Scale(input):
                    tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
                    if self.is_BN(tmp):
                        input = tmp
                inputs[0] = input
            i += 1
        params = node.layer.accuracy_param
        top_k = params.top_k
        axis = params.axis
        ignore_label = params.ignore_label
        # TODO(syf)
        assert axis == 1, 'PaddlePaddle can not support the situation when the axis is not 1.'
        assert not ignore_label >= 0, 'PaddlePaddle can not support the situation when the model has ignore label.'
        attr = {'k': top_k}
        node.fluid_code.add_layer("accuracy",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def TanH(self, node):
        assert len(
            node.inputs) == 1, 'The count of TanH node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        attr = {'name': string(node.layer_name)}
        node.fluid_code.add_layer("tanh",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Eltwise(self, node):
        assert len(
            node.inputs) == 2, 'The count of TanH node\'s input is not 2.'
        params = node.layer.eltwise_param
        mode = params.operation
        inputs = []
        input0 = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input0):
            tmp = self.graph.get_bottom_node(input0, idx=0, copy=True)
            if self.is_BN(tmp):
                input0 = tmp
        inputs.append(input0)
        input1 = self.graph.get_bottom_node(node, idx=1, copy=True)
        if self.is_Scale(input1):
            tmp = self.graph.get_bottom_node(input1, idx=0, copy=True)
            if self.is_BN(tmp):
                input1 = tmp
        inputs.append(input1)
        if mode == 0:
            inputs_dict = {}
            inputs_dict['x'] = inputs[0]
            inputs_dict['y'] = inputs[1]
            attr = {'act': None, 'name': string(node.layer_name)}
            node.fluid_code.add_layer("elementwise_mul",
                                      inputs=inputs_dict,
                                      output=node,
                                      param_attr=attr)
        elif mode == 1:
            if hasattr(params, 'coeff') and len(params.coeff) == 2:
                coeff = params.coeff
                input1_name = self.get_input_name(inputs[0])
                attr = {
                    'shape': [1],
                    'value': coeff[0],
                    'dtype': '{}.dtype'.format(input1_name)
                }
                node.fluid_code.add_layer("fill_constant",
                                          inputs=None,
                                          output=node.layer_name + '_const1',
                                          param_attr=attr)
                attr = {'act': None, 'name': string(node.layer_name + '_mul1')}
                node.fluid_code.add_layer("elementwise_mul",
                                          inputs=input1_name + ', ' +
                                          node.layer_name + '_const1',
                                          output=node.layer_name + '_mul1',
                                          param_attr=attr)
                input2_name = self.get_input_name(inputs[1])
                attr = {
                    'shape': [1],
                    'value': coeff[1],
                    'dtype': '{}.dtype'.format(input2_name)
                }
                node.fluid_code.add_layer("fill_constant",
                                          inputs=None,
                                          output=node.layer_name + '_const2',
                                          param_attr=attr)
                attr = {'act': None, 'name': string(node.layer_name + '_mul2')}
                node.fluid_code.add_layer("elementwise_mul",
                                          inputs=input2_name + ', ' +
                                          node.layer_name + '_const2',
                                          output=node.layer_name + '_mul2',
                                          param_attr=attr)

                attr = {'act': None, 'name': string(node.layer_name)}
                node.fluid_code.add_layer("elementwise_add",
                                          inputs='{}_mul1, {}_mul2'.format(
                                              node.layer_name, node.layer_name),
                                          output=node,
                                          param_attr=attr)
            else:
                inputs_dict = {}
                inputs_dict['x'] = inputs[0]
                inputs_dict['y'] = inputs[1]
                attr = {'act': None, 'name': string(node.layer_name)}
                node.fluid_code.add_layer("elementwise_add",
                                          inputs=inputs_dict,
                                          output=node,
                                          param_attr=attr)
        else:
            inputs_dict = {}
            inputs_dict['x'] = inputs[0]
            inputs_dict['y'] = inputs[1]
            attr = {'act': None, 'name': string(node.layer_name)}
            node.fluid_code.add_layer("elementwise_max",
                                      inputs=inputs_dict,
                                      output=node,
                                      param_attr=attr)

    def BatchNorm(self, node):
        assert len(node.inputs) == 1 and len(
            node.outputs
        ) == 1, 'The count of BatchNorm node\'s input and output is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        params = node.layer.batch_norm_param
        if hasattr(params, 'eps'):
            eps = params.eps
        else:
            eps = 1e-5
        assert len(node.data) == 3
        node.data = [np.squeeze(i) for i in node.data]
        mean, variance, scale = node.data
        # Prescale the stats
        scaling_factor = 1.0 / scale if scale != 0 else 0
        mean *= scaling_factor
        variance *= scaling_factor
        self.weights[node.layer_name + '_mean'] = mean
        self.weights[node.layer_name + '_variance'] = variance
        if self.graph.get_node(node.outputs[0]).layer_type == 'Scale':
            data = self.graph.get_node(node.outputs[0]).data
            self.weights[node.layer_name + '_scale'] = np.squeeze(data[0])
            self.weights[node.layer_name + '_offset'] = np.squeeze(data[1])
            attr = {
                'is_test': True,
                'param_attr': string(node.layer_name + '_scale'),
                'bias_attr': string(node.layer_name + '_offset'),
                'moving_mean_name': string(node.layer_name + '_mean'),
                'moving_variance_name': string(node.layer_name + '_variance'),
                'epsilon': eps,
                'name': string(node.layer_name)
            }
        else:
            attr = {
                'is_test': True,
                'param_attr': None,
                'bias_attr': None,
                'moving_mean_name': string(node.layer_name + '_mean'),
                'moving_variance_name': string(node.layer_name + '_variance'),
                'epsilon': eps,
                'name': string(node.layer_name)
            }
        node.fluid_code.add_layer("batch_norm",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Scale(self, node):
        assert len(
            node.outputs) == 1, 'The count of Scale node\'s output is not 1.'
        if len(node.inputs) == 1 and self.graph.get_node(
                node.inputs[0]).layer_type == 'BatchNorm':
            return
        else:
            self.weights[node.layer_name + '_scale'] = np.squeeze(nose.data[0])
            self.weights[node.layer_name + '_offset'] = np.squeeze(node.data[1])
            params = node.layer.scale_param
            axis = params.axis
            num_axes = params.num_axes
            assert num_axes == 1, "layer scale not support this num_axes[%d] now" % (
                num_axes)
            inputs = []
            if len(node.inputs) == 2:
                # for two tensor, here resets axis to 1. Maybe there is a bug for unkown case.
                axis = 1
                bias_shape = node.input_shape[0][axis:axis + num_axes]
                input0 = self.graph.get_bottom_node(node, idx=0, copy=True)
                if self.is_Scale(input0):
                    tmp = self.graph.get_bottom_node(input0, idx=0, copy=True)
                    if self.is_BN(tmp):
                        input0 = tmp
                input1 = self.graph.get_bottom_node(node, idx=1, copy=True)
                if self.is_Scale(input1):
                    tmp = self.graph.get_bottom_node(input1, idx=0, copy=True)
                    if self.is_BN(tmp):
                        input1 = tmp
                inputs.append(input0)
                inputs.append(input1)
                attr = {'axis': axis, 'name': string(node.layer_name + '_mul')}
                node.fluid_code.add_layer("elementwise_mul",
                                          inputs=inputs,
                                          output=node.layer_name + '_mul',
                                          param_attr=attr)
            else:
                bias_shape = node.input_shape[0][axis:axis + num_axes]
                input0 = self.graph.get_bottom_node(node, idx=0, copy=True)
                if self.is_Scale(input0):
                    tmp = self.graph.get_bottom_node(input0, idx=0, copy=True)
                    if self.is_BN(tmp):
                        input0 = tmp
                input0_name = self.get_input_name(input0)
                attr = {
                    'dtype': '{}.dtype'.formatr(input0_name),
                    'shape': bias_shape,
                    'name': string(node.layer_name + '_cparam1'),
                    'attr': string(node.layer_name + '_scale'),
                    'is_bias': True,
                    'default_initializer': 'Constant(value=1.0)'
                }
                node.fluid_code.add_layer("create_parameter",
                                          inputs=None,
                                          output=node,
                                          param_attr=attr)
                inputs.append(input0)
                inputs.append(node)
                attr = {'axis': axis, 'name': string(node.layer_name + '_mul')}
                node.fluid_code.add_layer("elementwise_mul",
                                          inputs=inputs,
                                          output=node.layer_name + '_mul',
                                          param_attr=attr)
            scale_shape = bias_shape
            input0_name = self.get_input_name(input0)
            attr = {
                'dtype': '{}.dtype'.formatr(input0_name),
                'shape': scale_shape,
                'name': string(node.layer_name + '_cparam2'),
                'attr': string(node.layer_name + '_offset'),
                'is_bias': True,
                'default_initializer': 'Constant(value=1.0)'
            }
            node.fluid_code.add_layer("create_parameter",
                                      inputs=None,
                                      output=node.layer_name + '_offset_param',
                                      param_attr=attr)
            attr = {'axis': axis, 'name': string(node.layer_name + '_add')}
            node.fluid_code.add_layer("elementwise_add",
                                      inputs='{}_mul, {}_offset_param'.format(
                                          node.layer_name, node.layer_name),
                                      output=node,
                                      param_attr=attr)

    def Reshape(self, node):
        assert len(node.inputs) == 1 and len(
            node.outputs
        ) == 1, 'The count of Reshape node\'s input and output is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        top_count = len(input.layer.top)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        is_inplace, = False if top_count == 1 else True
        output_shape = node.output_shape[0]
        attr = {
            'shape': output_shape,
            'inplace': is_inplace,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("reshape",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def ArgMax(self, node):
        assert len(node.inputs) == 1 and len(
            node.outputs
        ) == 1, 'The count of ArgMax node\'s input and output is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        input_shape = node.input_shape[0]
        params = node.layer.argmax_param
        out_max_val = params.out_max_val if hasattr(params,
                                                    out_max_val) else False
        top_k = params.top_k if hasattr(params, top_k) else 1
        axis = parmas.axis if hasattr(params, axis) else -1
        if axis < 0:
            axis += len(input_shape)
        if out_max_val is True:
            attr = {'k': top_k, 'name': string(node.layer_name + '_topk')}
            node.fluid_code.add_layer("topk",
                                      inputs=input,
                                      output='{}_topk_var, {}_index_var'.format(
                                          node.layer_name, node.layer_name),
                                      param_attr=attr)
            attr = {'dtype': '{}_topk_var.dtype'.format(node.layer_name)}
            node.fluid_code.add_layer(
                "cast",
                inputs='{}_index_var'.format(node.layer_name),
                output='{}_index_var'.format(node.layer_name),
                param_attr=attr)
            attr = {'axis': axis, 'name': string(node.layer_name)}
            node.fluid_code.add_layer("concat",
                                      inputs='{}_topk_var, {}_index_var'.format(
                                          node.layer_name, node.layer_name),
                                      output=node,
                                      param_attr=attr)
        else:
            attr = {'k': top_k, 'name': string(node.layer_name)}
            node.fluid_code.add_layer("topk",
                                      inputs=input,
                                      output='_, {}'.format(node.layer_name),
                                      param_attr=attr)

    def Axpy(self, node):
        assert len(
            node.inputs) == 3, 'The count of Axpy node\'s input is not 3.'
        alpha = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(alpha):
            tmp = self.graph.get_bottom_node(alpha, idx=0, copy=True)
            if self.is_BN(tmp):
                alpha = tmp
        x = self.graph.get_bottom_node(node, idx=1, copy=True)
        if self.is_Scale(x):
            tmp = self.graph.get_bottom_node(x, idx=0, copy=True)
            if self.is_BN(tmp):
                x = tmp
        y = self.graph.get_bottom_node(node, idx=2, copy=True)
        if self.is_Scale(y):
            tmp = self.graph.get_bottom_node(y, idx=0, copy=True)
            if self.is_BN(tmp):
                y = tmp
        attr = {'axis': 0, 'name': string(node.layer_name + '_mul')}
        node.fluid_code.add_layer("elementwise_mul",
                                  inputs={
                                      'x': alpha,
                                      'y': x
                                  },
                                  output=node,
                                  param_attr=attr)
        attr = {'name': string(node.layer_name + '_add')}
        node.fluid_code.add_layer("elementwise_add",
                                  inputs={
                                      'x': node,
                                      'y': y
                                  },
                                  output=node,
                                  param_attr=attr)

    def Crop(self, node):
        assert len(
            node.inputs) == 2, 'The count of Crop node\'s input is not 2.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        example = self.graph.get_bottom_node(node, idx=1, copy=True)
        if self.is_Scale(example):
            tmp = self.graph.get_bottom_node(example, idx=0, copy=True)
            if self.is_BN(tmp):
                example = tmp
        params = node.layer.crop_param
        axis = parmas.axis
        input_shape = node.input_shape[0]
        if axis < 0:
            axis += len(input_shape)
        offset_real = [0] * len(input_shape)
        if hasattr(params, offset):
            offset = list(params.offset)
            assert (len(input_shape) - axis) == len(
                offset), "invalid offset[%s] in crop layer" % (str(offset))
            offset_real = [0] * axis + offset
        attr = {'offsets': offset_real, 'name': string(node.layer_name)}
        node.fluid_code.add_layer("crop",
                                  inputs={
                                      'x': input,
                                      'y': example
                                  },
                                  output=node,
                                  param_attr=attr)

    def DetectionOutput(self, node):
        assert len(
            node.inputs
        ) == 3, 'The count of DetectionOutput node\'s input is not 3.'
        mbox_loc = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(mbox_loc):
            tmp = self.graph.get_bottom_node(mbox_loc, idx=0, copy=True)
            if self.is_BN(tmp):
                mbox_loc = tmp
        mbox_conf_flatten = self.graph.get_bottom_node(node, idx=1, copy=True)
        if self.is_Scale(mbox_conf_flatten):
            tmp = self.graph.get_bottom_node(mbox_conf_flatten,
                                             idx=0,
                                             copy=True)
            if self.is_BN(tmp):
                mbox_conf_flatten = tmp
        mbox_priorbox = self.graph.get_bottom_node(node, idx=2, copy=True)
        if self.is_Scale(mbox_priorbox):
            tmp = self.graph.get_bottom_node(mbox_priorbox, idx=0, copy=True)
            if self.is_BN(tmp):
                mbox_priorbox = tmp
        params = node.layer.detection_output_param
        nms_threshold = 0.3
        top_k = 10
        eta = 1.0
        if hasattr(params, 'nms_param'):
            nms_threshold = getattr(params.nms_param, 'nms_threshold', 0.3)
            top_k = getattr(params.nms_param, 'top_k', 10)
            eta = getattr(params.nms_param, 'eta', 1.0)
        background_label = getattr(params, 'background_label_id', 0)
        share_location = getattr(params, 'share_location', True)
        keep_top_k = getattr(params, 'keep_top_k', 100)
        confidence_threshold = getattr(params, 'confidence_threshold', 0.1)
        attr = {
            'num_or_sections': 2,
            'dim': 1,
            'name': string(node.layer_name + '_split')
        }
        node.fluid_code.add_layer("split",
                                  inputs=mbox_priorbox,
                                  output='mbox_priorbox_list',
                                  param_attr=attr)
        node.fluid_code.add_note('pb = mbox_priorbox_list[0]')
        node.fluid_code.add_note('pbv = mbox_priorbox_list[1]')
        attr = {'shape': [-1, 4], 'name': string(node.layer_name + '_reshape1')}
        node.fluid_code.add_layer("reshape",
                                  inputs='pb',
                                  output='pb',
                                  param_attr=attr)
        attr = {'shape': [-1, 4], 'name': string(node.layer_name + '_reshape2')}
        node.fluid_code.add_layer("reshape",
                                  inputs='pbv',
                                  output='pbv',
                                  param_attr=attr)
        # TODO(syf): need chaeck
        attr = {
            'shape': [-1, node.input_shape[1][1], 4],
            'name': string(node.layer_name + '_reshape3')
        }
        node.fluid_code.add_layer("reshape",
                                  inputs=mbox_loc,
                                  output='mbox_loc',
                                  param_attr=attr)
        attr = {
            'background_label': background_label,
            'nms_threshold': nms_threshold,
            'nms_top_k': top_k,
            'keep_top_k': keep_top_k,
            'score_threshold': confidence_threshold,
            'nms_eta': eta
        }
        inputs_str = get_input_name(mbox_conf_flatten) + ', mbox_loc, pb, pbv'
        node.fluid_code.add_layer("detection_output",
                                  inputs=inputs_str,
                                  output=node,
                                  param_attr=attr)

    def Flatten(self, noed):
        assert len(
            node.inputs
        ) == 1, 'The count of DetectionOutput node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        shape = node.output_shape[0]
        attr = {'shape': shape, 'name': string(node.layer_name)}
        node.fluid_code.add_layer("reshape",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Normalize(self, node):
        assert len(
            node.inputs) == 1, 'The count of Normalize node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        params = node.layer.norm_param
        across_spatial = params.across_spatial
        channel_shared = params.channel_shared
        assert across_spatial == False, "Only support across_spatial == False for Normalize"
        attr = {'axis': 1, 'name': string(node.layer_name + '_l2')}
        node.fluid_code.add_layer("l2_normalize",
                                  inputs=input,
                                  output=node.layer_name + '_l2',
                                  param_attr=attr)
        input_name = self.get_input_name(input)
        data = node.data
        data = self.adjust_parameters(node, data)
        self.weights[node.layer_name + '_scale'] = data[0]
        node.fluid_code.add_note(
            '{}_scale_attr = ParamAttr(name=\'{}\')'.format(
                node.layer_name, node.layer_name + '_scale'))
        attr = {
            'shape': [1] if channel_shared else [node.input_shape[0][1]],
            'dtype': '{}.dtype'.format(input_name),
            'attr': '{}_scale_attr'.format(node.layer_name),
            'name': string(node.layer_name + '_param')
        }
        node.fluid_code.add_layer("create_parameter",
                                  inputs=None,
                                  output=node.layer_name + '_scale_param',
                                  param_attr=attr)
        attr = {
            'axis': -1 if channel_shared else 1,
            'name': string(node.layer_name + '_mul')
        }
        node.fluid_code.add_layer("elementwise_mul",
                                  inputs=node.layer_name + '_l2, ' +
                                  node.layer_name + '_scale_param',
                                  output=node,
                                  param_attr=attr)

    def Permute(self, node):
        assert len(
            node.inputs) == 1, 'The count of Permute node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        params = node.layer.permute_param
        order = list(params.order)
        attr = {'order': order, 'name': string(node.layer_name)}
        node.fluid_code.add_layer("transpose",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def Power(self, node):
        assert len(
            node.inputs) == 1, 'The count of Permute node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        params = node.layer.power_param
        power = params.power
        scale = params.scale
        shift = params.shift
        attr = {
            'scale': scale,
            'bias': shift,
            'bias_after_scale': True,
            'name': string(node.layer_name + '_scale')
        }
        node.fluid_code.add_layer("scale",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)
        attr = {'factor': power, 'name': string(node.layer_name)}
        node.fluid_code.add_layer("pow",
                                  inputs=node,
                                  output=node,
                                  param_attr=attr)

    def PriorBox(self, node):
        assert len(
            node.inputs) == 2, 'The count of PriorBox node\'s input is not 2.'
        input1 = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input1):
            tmp = self.graph.get_bottom_node(input1, idx=0, copy=True)
            if self.is_BN(tmp):
                input1 = tmp
        input2 = self.graph.get_bottom_node(node, idx=1, copy=True)
        if self.is_Scale(input2):
            tmp = self.graph.get_bottom_node(input2, idx=0, copy=True)
            if self.is_BN(tmp):
                input2 = tmp
        input_dict = {'input': input1, 'image': input2}
        params = node.layer.prior_box_param
        step = getattr(params, 'step', 0.0)
        offset = getattr(params, 'offset', 0.5)
        min_size = list(params.min_size)
        max_size = list(params.max_size)
        aspect_ratio = list(params.aspect_ratio)
        flip = getattr(params, 'flip', False)
        clip = getattr(params, 'clip', False)
        variance = list(getattr(params, 'variance', [0.1, 0.1, 0.2, 0.2]))
        steps = tuple(step) if type(step) is list or type(step) is tuple else (
            step, step)
        attr = {
            'min_sizes': min_size,
            'max_sizes': max_size,
            'aspect_ratios': aspect_ratio,
            'variance': variance,
            'flip': flip,
            'clip': clip,
            'step': steps,
            'offset': offset,
            'min_max_aspect_ratios_order': True,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("prior_box",
                                  inputs=input_dict,
                                  output='{}_box, {}_var'.format(
                                      node.layer_name, node.layer_name),
                                  param_attr=attr)
        attr = {
            'shape': [1, 1, -1],
        }
        node.fluid_code.add_layer("reshape",
                                  inputs='{}_box'.format(node.layer_name),
                                  output='{}_box'.format(node.layer_name),
                                  param_attr=attr)
        attr = {
            'shape': [1, 1, -1],
        }
        node.fluid_code.add_layer("reshape",
                                  inputs='{}_var'.format(node.layer_name),
                                  output='{}_var'.format(node.layer_name),
                                  param_attr=attr)
        attr = {'axis': 1, 'name': string(node.layer_name + '_concat')}
        node.fluid_code.add_layer("concat",
                                  inputs='[{}_box, {}_var]'.format(
                                      node.layer_name, node.layer_name),
                                  output=node,
                                  param_attr=attr)

    def Reduction(self, node):
        assert len(
            node.inputs) == 1, 'The count of Reduction node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        params = node.layer.reduction_param
        operation = params.operation
        axis = params.axis
        coeff = params.coeff
        assert operation >= 1 and operation <= 4, "reduction reduction [%s] error" % (
            operation)
        input_len = len(node.input_shape[0])
        if axis < 0:
            axis += input_len + 1
        dim = list(range(input_len))
        if operation == 1:  ## operation = SUM
            attr = {
                'dim': dim[axis:],
                'keep_dim': False,
                'name': string(node.layer_name)
            }
            node.fluid_code.add_layer("reduce_sum",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
        elif operation == 2:  ## operation = ASUM
            attr = {'name': string(node.layer_name + '_abs')}
            node.fluid_code.add_layer("abs",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            attr = {
                'dim': dim[axis:],
                'keep_dim': False,
                'name': string(node.layer_name)
            }
            node.fluid_code.add_layer("reduce_sum",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)
        elif operation == 3:  ## operation = SUMSQ
            attr = {'factor': 2.0, 'name': string(node.layer_name + '_pow')}
            node.fluid_code.add_layer("pow",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            attr = {
                'dim': dim[axis:],
                'keep_dim': False,
                'name': string(node.layer_name)
            }
            node.fluid_code.add_layer("reduce_sum",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)
        else:  ## operation = MEAN
            attr = {
                'dim': dim[axis:],
                'keep_dim': False,
                'name': string(node.layer_name)
            }
            node.fluid_code.add_layer("reduce_mean",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)
        attr = {'scale': coeff}
        node.fluid_code.add_layer("scale",
                                  inputs=node,
                                  output=node,
                                  param_attr=attr)

    def ROIPooling(self, node):
        assert len(
            node.inputs) == 2, 'The count of ROIPooling node\'s input is not 2.'
        input1 = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input1):
            tmp = self.graph.get_bottom_node(input1, idx=0, copy=True)
            if self.is_BN(tmp):
                input1 = tmp
        input2 = self.graph.get_bottom_node(node, idx=1, copy=True)
        if self.is_Scale(input2):
            tmp = self.graph.get_bottom_node(input2, idx=0, copy=True)
            if self.is_BN(tmp):
                input2 = tmp
        attr = {'axes': [1], 'starts': [1], 'ends': [5]}
        node.fluid_code.add_layer("slice",
                                  inputs=input2,
                                  output=input2,
                                  param_attr=attr)
        input_dict = {'input': input1, 'rois': input2}
        params = node.layer.roi_pooling_param
        attr = {
            'pooled_w': params.pooled_w,
            'pooled_h': params.pooled_h,
            'spatial_scale': params.spatial_scale,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("roi_pool",
                                  inputs=input_dict,
                                  output=node,
                                  param_attr=attr)

    def Select(self, node):
        assert len(
            node.inputs) == 1, 'The count of Select node\'s input is not 2.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        if self.is_Scale(input):
            tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
            if self.is_BN(tmp):
                input = tmp
        params = node.layer.select_param
        slice_point = list(params.slice_point)
        axis = params.axis
        maxint32 = 2147483647
        slice_point = [0] + slice_point
        slice_point.append(maxint32)
        i = 0
        node.fluid_code.add_note('{} = []'.format(node.layer_name))
        for i in range(len(slice_point)):
            attr = {
                'axes': [axis],
                'starts': [slice_point[i]],
                'ends': [slice_point[i + 1]],
                'name': string(node.layer_name + '_' + str(i))
            }
            node.fluid_code.add_layer("slice",
                                      inputs=input,
                                      output=string(node.layer_name + '_' +
                                                    str(i)),
                                      param_attr=attr)
            node.fluid_code.add_note('{}.append({})'.format(
                node.layer_name, node.layer_name + '_' + str(i)))
            if i == len(slice_point) - 2:
                break
