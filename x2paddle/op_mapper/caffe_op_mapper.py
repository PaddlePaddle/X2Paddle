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
from x2paddle.op_mapper import caffe_shape
from x2paddle.op_mapper.caffe_custom_layer import *


class CaffeOpMapper(OpMapper):
    directly_map_ops = {
        'AbsVal': 'abs',
        'Sigmoid': 'sigmoid',
        'TanH': 'tanh',
    }

    def __init__(self, decoder):
        super(CaffeOpMapper, self).__init__()
        self.graph = decoder.caffe_graph
        self.weights = dict()
        resolver = decoder.resolver
        self.used_custom_layers = {}

        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node.layer_type == 'DepthwiseConvolution':
                node.layer_type = 'ConvolutionDepthwise'
            op = node.layer_type
            if hasattr(self, op):
                self.set_node_shape(node)
                func = getattr(self, op)
                func(node)
            elif op in custom_layers:
                self.set_node_shape(node, is_fluid_op=False)
                self.deal_custom_layer(node)
            elif op in self.directly_map_ops:
                self.set_node_shape(node)
                self.directly_map(node)
            else:
                raise Exception(
                    "The op {} in model is not supported yet.".format(op))

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self, op) and op not in custom_layers:
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            print("There are {} ops not supported yet, list as below".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print(op)
            return False

    def set_node_shape(self, node, is_fluid_op=True):
        inputs = node.inputs
        input_shape = []
        for i, nm in enumerate(inputs):
            last_node = self.graph.get_node(nm)
            tmp = node.layer.bottom[i]
            idx = list(last_node.layer.top).index(tmp)
            input_shape.append(last_node.output_shape[idx])

        node.input_shape = input_shape

        func_name = 'shape_' + node.layer_type.lower()
        if is_fluid_op:
            node.output_shape = getattr(caffe_shape, func_name)(node.layer,
                                                                input_shape)
        else:
            node.output_shape = compute_output_shape(node)

    def adjust_parameters(self, node):
        data = node.data
        # When using the protobuf-backend, each parameter initially has four dimensions.
        # In certain cases (like FC layers), we want to eliminate the singleton dimensions.
        # This implementation takes care of the common cases. However, it does leave the
        # potential for future issues.
        # The Caffe-backend does not suffer from this problem.
        data = list(data)

        squeeze_indices = [1]  # Squeeze biases.
        if node.layer_type == 'InnerProduct':
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
        return data

    def get_kernel_parameters(self, kind, params):
        assert kind in ['Convolution', 'Pooling', 'Deconvolution']
        [k_h, k_w] = [1, 1]
        if isinstance(params.kernel_size, numbers.Number):
            [k_h, k_w] = [params.kernel_size] * 2
        elif len(params.kernel_size) > 0:
            k_h = params.kernel_h if params.kernel_h > 0 else params.kernel_size[
                0]
            k_w = params.kernel_w if params.kernel_w > 0 else params.kernel_size[
                len(params.kernel_size) - 1]
        elif params.kernel_h > 0 or params.kernel_w > 0:
            k_h = params.kernel_h
            k_w = params.kernel_w
        [s_h, s_w] = [1, 1]
        if isinstance(params.stride, numbers.Number):
            [s_h, s_w] = [params.stride] * 2
        elif len(params.stride) > 0:
            s_h = params.stride_h if params.stride_h > 0 else params.stride[0]
            s_w = params.stride_w if params.stride_w > 0 else params.stride[len(
                params.stride) - 1]
        elif params.stride_h > 0 or params.stride_w > 0:
            s_h = params.stride_h
            s_w = params.stride_w
        [p_h, p_w] = [0, 0]
        if isinstance(params.pad, numbers.Number):
            [p_h, p_w] = [params.pad] * 2
        elif len(params.pad) > 0:
            p_h = params.pad_h if params.pad_h > 0 else params.pad[0]
            p_w = params.pad_w if params.pad_w > 0 else params.pad[len(
                params.pad) - 1]
        elif params.pad_h > 0 or params.pad_w > 0:
            p_h = params.pad_h
            p_w = params.pad_w
        dila_h = dila_w = 1
        group = 1
        c_o = 1
        if kind in ['Convolution', 'Deconvolution']:
            c_o = params.num_output
            dila_len = len(params.dilation)
            if dila_len == 2:
                dila_h = params.dilation[0]
                dila_w = params.dilation[1]
            elif dila_len == 1:
                dila_h = dila_w = params.dilation[0]
            else:
                assert dila_len == 0, "invalid length[%s] of dilation in convolution" % (
                    dila_len)
        if kind in ['Convolution', 'Deconvolution']:
            group = params.group
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

    def Input(self, node):
        shape = list(node.layer.input_param.shape[0].dim)[1:]
        dtype = 'float32'
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "data", inputs=None, output=node, param_attr=attr)

    def MemoryData(self, node):
        # TODO(syf): Paddlepaddle can't fully support
        shape = node.output_shape[0][1:]
        dtype = 'float32'
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "data", inputs=None, output=node.layer_name + '0', param_attr=attr)
        node.fluid_code.add_note('{} = [{}]'.format(node.layer_name,
                                                    node.layer_name + '0'))

    def Convolution(self, node):
        data = node.data
        params = node.layer.convolution_param
        channel, kernel, stride, pad, dilation, group = self.get_kernel_parameters(
            node.layer_type, params)
        if data is None:
            data = []
            print(
                'The parameter of {} (type is {}) is not set. So we set the parameters as 0'
                .format(node.layer_name, node.layer_type))
            input_c = node.input_shape[0][1]
            output_c = channel
            data.append(
                np.zeros([output_c, input_c, kernel[0], kernel[1]]).astype(
                    'float32'))
            data.append(np.zeros([output_c, ])).astype('float32')
        else:
            data = self.adjust_parameters(node)
        self.weights[node.layer_name + '_weights'] = data[0]
        if len(data) == 2:
            self.weights[node.layer_name + '_bias'] = data[1]
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
            'bias_attr': False
            if len(data) == 1 else string(node.layer_name + '_bias'),
        }
        node.fluid_code.add_layer(
            "conv2d", inputs=input, output=node, param_attr=attr)

    def Deconvolution(self, node):
        data = node.data
        params = node.layer.convolution_param
        channel, kernel, stride, pad, dilation, group = self.get_kernel_parameters(
            node.layer_type, params)
        if data is None:
            data = []
            print(
                'The parameter of {} (type is {}) is not set. So we set the parameters as 0'
                .format(node.layer_name, node.layer_type))
            input_c = node.input_shape[0][1]
            output_c = channel
            data.append(
                np.zeros([output_c, input_c, kernel[0], kernel[1]]).astype(
                    'float32'))
            data.append(np.zeros([output_c, ]).astype('float32'))
        else:
            data = self.adjust_parameters(node)
        self.weights[node.layer_name + '_weights'] = data[0]
        if len(data) == 2:
            self.weights[node.layer_name + '_bias'] = data[1]
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
            'bias_attr': False
            if len(data) == 1 else string(node.layer_name + '_bias')
        }
        node.fluid_code.add_layer(
            "conv2d_transpose", inputs=input, output=node, param_attr=attr)

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
        attr = {
            'pool_size': kernel,
            'pool_stride': stride,
            'pool_padding': pad,
            'ceil_mode': ceil_mode,
            'pool_type': string(pool_type),
            'exclusive': False,
            'global_pooling': global_pool,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "pool2d", inputs=input, output=node, param_attr=attr)

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
            'k': params.k,
            'alpha': alpha,
            'beta': params.beta,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "lrn", inputs=input, output=node, param_attr=attr)

    def InnerProduct(self, node):
        data = node.data
        params = node.layer.inner_product_param
        if data is None:
            print(
                'The parameter of {} (type is {}) is not set. So we set the parameters as 0.'
                .format(node.layer_name, node.layer_type))
            input_c = node.input_shape[0][1]
            output_c = params.num_output
            data = []
            data.append(
                np.zeros([input_c, output_c]).astype('float32').astype(
                    'float32'))
            data.append(
                np.zeros([output_c]).astype('float32').astype('float32'))
        else:
            data = self.adjust_parameters(node)
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
        #params = node.layer.inner_product_param
        assert params.axis == 1
        assert params.bias_term == True
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        attr = {
            'size': params.num_output,
            'name': string(node.layer_name),
            'act': None,
            'param_attr': string(node.layer_name + '_weights'),
            'bias_attr': False
            if len(data) == 1 else string(node.layer_name + '_bias')
        }
        node.fluid_code.add_layer(
            "fc", inputs=input, output=node, param_attr=attr)

    def Softmax(self, node):
        assert len(
            node.inputs) == 1, 'The count of Softmax node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        params = node.layer.softmax_param
        axis = params.axis
        shape = node.input_shape[0]
        dims = len(shape)
        axis = axis + dims if axis < 0 else axis
        attr = {'axis': axis, 'name': string(node.layer_name + '_softmax')}
        node.fluid_code.add_layer(
            "softmax", inputs=input, output=node, param_attr=attr)

    def Slice(self, node):
        assert len(
            node.inputs) == 1, 'The count of Slice node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        top_len = len(node.layer.top)
        params = node.layer.slice_param
        axis = params.axis
        slice_dim = params.slice_dim
        if slice_dim != 1 and axis == 1:
            axis = slice_dim
        output_shape = node.output_shape
        sections_list = []
        for s in output_shape:
            sections_list.append(s[axis])
        attr = {
            'num_or_sections': sections_list,
            'dim': axis,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "split", inputs=input, output=node.layer_name, param_attr=attr)

    def Concat(self, node):
        assert len(
            node.inputs
        ) >= 1, 'The count of Concat node\'s input is not more than 1.'
        inputs = []
        for i in range(len(node.inputs)):
            input = self.graph.get_bottom_node(node, idx=i, copy=True)
            inputs.append(input)
        params = node.layer.concat_param
        axis = params.axis
        attr = {'axis': axis, 'name': string(node.layer_name)}
        node.fluid_code.add_layer(
            "concat", inputs=inputs, output=node, param_attr=attr)

    def ReLU(self, node):
        """

        :param node:
        :return:
        """
        assert len(
            node.inputs) == 1, 'The count of ReLU node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)

        params = node.layer.relu_param
        if params.HasField('negative_slope') and params.negative_slope != 0:
            negative_slope = float(params.negative_slope)

            attr = {'alpha': negative_slope}
            node.fluid_code.add_layer(
                'leaky_relu', inputs=input, output=node, param_attr=attr)
        else:
            node.fluid_code.add_layer('relu', inputs=input, output=node)

    def PReLU(self, node):
        assert len(
            node.inputs) == 1, 'The count of PReLU node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
            'mode': string(mode),
            'param_attr': string(node.layer_name + '_weights'),
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "prelu", inputs=input, output=node, param_attr=attr)

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
                inputs[1] = input
            else:
                input = self.graph.get_bottom_node(node, idx=i, copy=True)
                inputs[0] = input
            i += 1
        params = node.layer.accuracy_param
        top_k = params.top_k
        axis = params.axis
        ignore_label = params.ignore_label
        assert axis == 1, 'PaddlePaddle can not support the situation when the axis is not 1.'
        assert not ignore_label >= 0, 'PaddlePaddle can not support the situation when the model has ignore label.'
        attr = {'k': top_k}
        node.fluid_code.add_layer(
            "accuracy", inputs=inputs, output=node, param_attr=attr)

    def Eltwise(self, node):
        assert len(
            node.inputs) == 2, 'The count of TanH node\'s input is not 2.'
        params = node.layer.eltwise_param
        mode = params.operation
        inputs = []
        input0 = self.graph.get_bottom_node(node, idx=0, copy=True)
        inputs.append(input0)
        input1 = self.graph.get_bottom_node(node, idx=1, copy=True)
        inputs.append(input1)
        if mode == 0:
            inputs_dict = {}
            inputs_dict['x'] = inputs[0]
            inputs_dict['y'] = inputs[1]
            attr = {'act': None, 'name': string(node.layer_name)}
            node.fluid_code.add_layer(
                "elementwise_mul",
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
                node.fluid_code.add_layer(
                    "fill_constant",
                    inputs=None,
                    output=node.layer_name + '_const1',
                    param_attr=attr)
                attr = {'act': None, 'name': string(node.layer_name + '_mul1')}
                node.fluid_code.add_layer(
                    "elementwise_mul",
                    inputs=input1_name + ', ' + node.layer_name + '_const1',
                    output=node.layer_name + '_mul1',
                    param_attr=attr)
                input2_name = self.get_input_name(inputs[1])
                attr = {
                    'shape': [1],
                    'value': coeff[1],
                    'dtype': '{}.dtype'.format(input2_name)
                }
                node.fluid_code.add_layer(
                    "fill_constant",
                    inputs=None,
                    output=node.layer_name + '_const2',
                    param_attr=attr)
                attr = {'act': None, 'name': string(node.layer_name + '_mul2')}
                node.fluid_code.add_layer(
                    "elementwise_mul",
                    inputs=input2_name + ', ' + node.layer_name + '_const2',
                    output=node.layer_name + '_mul2',
                    param_attr=attr)

                attr = {'act': None, 'name': string(node.layer_name)}
                node.fluid_code.add_layer(
                    "elementwise_add",
                    inputs='{}_mul1, {}_mul2'.format(node.layer_name,
                                                     node.layer_name),
                    output=node,
                    param_attr=attr)
            else:
                inputs_dict = {}
                inputs_dict['x'] = inputs[0]
                inputs_dict['y'] = inputs[1]
                attr = {'act': None, 'name': string(node.layer_name)}
                node.fluid_code.add_layer(
                    "elementwise_add",
                    inputs=inputs_dict,
                    output=node,
                    param_attr=attr)
        else:
            inputs_dict = {}
            inputs_dict['x'] = inputs[0]
            inputs_dict['y'] = inputs[1]
            attr = {'act': None, 'name': string(node.layer_name)}
            node.fluid_code.add_layer(
                "elementwise_max",
                inputs=inputs_dict,
                output=node,
                param_attr=attr)

    def BatchNorm(self, node):
        assert len(
            node.inputs) == 1, 'The count of BatchNorm node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        params = node.layer.batch_norm_param
        if hasattr(params, 'eps'):
            eps = params.eps
        else:
            eps = 1e-5
        if node.data is None or len(node.data) != 3:
            print(
                'The parameter of {} (type is {}) is not set. So we set the parameters as 0'
                .format(node.layer_name, node.layer_type))
            input_c = node.input_shape[0][1]
            mean = np.zeros([input_c, ]).astype('float32')
            variance = np.zeros([input_c, ]).astype('float32')
            scale = 0
        else:

            node.data = [np.squeeze(i).astype('float32') for i in node.data]
            mean, variance, scale = node.data
        # Prescale the stats
        scaling_factor = 1.0 / scale if scale != 0 else 0
        mean *= scaling_factor
        variance *= scaling_factor
        self.weights[node.layer_name + '_mean'] = mean
        self.weights[node.layer_name + '_variance'] = variance
        attr = {
            'is_test': True,
            'param_attr': None,
            'bias_attr': None,
            'moving_mean_name': string(node.layer_name + '_mean'),
            'moving_variance_name': string(node.layer_name + '_variance'),
            'epsilon': eps,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "batch_norm", inputs=input, output=node, param_attr=attr)

    def Scale(self, node):
        if node.data is None:
            print(
                'The parameter of {} (type is {}) is not set. So we set the parameters as 0'
                .format(node.layer_name, node.layer_type))
            input_c = node.input_shape[0][1]
            self.weights[node.layer_name + '_scale'] = np.zeros([
                input_c,
            ]).astype('float32')
            self.weights[node.layer_name + '_offset'] = np.zeros([
                input_c,
            ]).astype('float32')
        else:
            self.weights[node.layer_name + '_scale'] = np.squeeze(node.data[
                0]).astype('float32')
            self.weights[node.layer_name + '_offset'] = np.squeeze(node.data[
                1]).astype('float32')
        params = node.layer.scale_param
        axis = params.axis
        num_axes = params.num_axes
        inputs = []
        if len(node.inputs) == 2:
            # for two tensor, here resets axis to 1. Maybe there is a bug for unkown case.
            axis = 1
            bias_shape = node.input_shape[0][axis:axis + num_axes]
            input0 = self.graph.get_bottom_node(node, idx=0, copy=True)
            input1 = self.graph.get_bottom_node(node, idx=1, copy=True)
            inputs_dict = {}
            inputs_dict['x'] = input0
            inputs_dict['y'] = input1
            attr = {'axis': axis, 'name': string(node.layer_name + '_mul')}
            node.fluid_code.add_layer(
                "elementwise_mul",
                inputs=inputs_dict,
                output=node.layer_name + '_mul',
                param_attr=attr)
        else:
            bias_shape = node.input_shape[0][axis:axis + num_axes]
            input0 = self.graph.get_bottom_node(node, idx=0, copy=True)
            input0_name = self.get_input_name(input0)
            attr = {
                'dtype': '{}.dtype'.format(input0_name),
                'shape': bias_shape,
                'name': string(node.layer_name + '_cparam1'),
                'attr': string(node.layer_name + '_scale'),
                'is_bias': True,
                'default_initializer': 'Constant(value=1.0)'
            }
            node.fluid_code.add_layer(
                "create_parameter", inputs=None, output=node, param_attr=attr)
            inputs_dict = {}
            inputs_dict['x'] = input0
            inputs_dict['y'] = node
            attr = {'axis': axis, 'name': string(node.layer_name + '_mul')}
            node.fluid_code.add_layer(
                "elementwise_mul",
                inputs=inputs_dict,
                output=node.layer_name + '_mul',
                param_attr=attr)
        scale_shape = bias_shape
        input0_name = self.get_input_name(input0)
        attr = {
            'dtype': '{}.dtype'.format(input0_name),
            'shape': scale_shape,
            'name': string(node.layer_name + '_cparam2'),
            'attr': string(node.layer_name + '_offset'),
            'is_bias': True,
            'default_initializer': 'Constant(value=1.0)'
        }
        node.fluid_code.add_layer(
            "create_parameter",
            inputs=None,
            output=node.layer_name + '_offset_param',
            param_attr=attr)
        attr = {'axis': axis, 'name': string(node.layer_name + '_add')}
        node.fluid_code.add_layer(
            "elementwise_add",
            inputs='{}_mul, {}_offset_param'.format(node.layer_name,
                                                    node.layer_name),
            output=node,
            param_attr=attr)

    def Reshape(self, node):
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        top_count = len(input.layer.top)
        is_inplace = False if top_count == 1 else True
        output_shape = node.output_shape[0]
        attr = {
            'shape': output_shape,
            'inplace': is_inplace,
            'act': None,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "reshape", inputs=input, output=node, param_attr=attr)

    def ArgMax(self, node):
        assert len(node.inputs) == 1 and len(
            node.outputs
        ) == 1, 'The count of ArgMax node\'s input and output is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
            node.fluid_code.add_layer(
                "topk",
                inputs=input,
                output='{}_topk_var, {}_index_var'.format(node.layer_name,
                                                          node.layer_name),
                param_attr=attr)
            attr = {'dtype': '{}_topk_var.dtype'.format(node.layer_name)}
            node.fluid_code.add_layer(
                "cast",
                inputs='{}_index_var'.format(node.layer_name),
                output='{}_index_var'.format(node.layer_name),
                param_attr=attr)
            attr = {'axis': axis, 'name': string(node.layer_name)}
            node.fluid_code.add_layer(
                "concat",
                inputs='{}_topk_var, {}_index_var'.format(node.layer_name,
                                                          node.layer_name),
                output=node,
                param_attr=attr)
        else:
            attr = {'k': top_k, 'name': string(node.layer_name)}
            node.fluid_code.add_layer(
                "topk",
                inputs=input,
                output='_, {}'.format(node.layer_name),
                param_attr=attr)

    def Crop(self, node):
        assert len(
            node.inputs) == 2, 'The count of Crop node\'s input is not 2.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        example = self.graph.get_bottom_node(node, idx=1, copy=True)
        params = node.layer.crop_param
        axis = params.axis
        input_shape = node.input_shape[0]
        if axis < 0:
            axis += len(input_shape)
        offset_real = [0] * len(input_shape)
        if hasattr(params, "offset") and len(params.offset) > 0:
            offset = list(params.offset)
            assert (len(input_shape) - axis
                    ) == len(offset), "invalid offset[%s] in crop layer" % (
                        str(offset))
            offset_real = [0] * axis + offset
        attr = {'offsets': list(offset_real), 'name': string(node.layer_name)}
        node.fluid_code.add_layer(
            "crop",
            inputs={'x': input,
                    'shape': node.input_shape[1]},
            output=node,
            param_attr=attr)

    def Flatten(self, node):
        assert len(
            node.
            inputs) == 1, 'The count of DetectionOutput node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        shape = node.output_shape[0]
        attr = {'shape': shape, 'name': string(node.layer_name)}
        node.fluid_code.add_layer(
            "reshape", inputs=input, output=node, param_attr=attr)

    def Power(self, node):
        assert len(
            node.inputs) == 1, 'The count of Permute node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
        node.fluid_code.add_layer(
            "scale", inputs=input, output=node, param_attr=attr)
        attr = {'factor': power, 'name': string(node.layer_name)}
        node.fluid_code.add_layer(
            "pow", inputs=node, output=node, param_attr=attr)

    def Reduction(self, node):
        assert len(
            node.inputs) == 1, 'The count of Reduction node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
            node.fluid_code.add_layer(
                "reduce_sum", inputs=input, output=node, param_attr=attr)
        elif operation == 2:  ## operation = ASUM
            attr = {'name': string(node.layer_name + '_abs')}
            node.fluid_code.add_layer(
                "abs", inputs=input, output=node, param_attr=attr)
            attr = {
                'dim': dim[axis:],
                'keep_dim': False,
                'name': string(node.layer_name)
            }
            node.fluid_code.add_layer(
                "reduce_sum", inputs=node, output=node, param_attr=attr)
        elif operation == 3:  ## operation = SUMSQ
            attr = {'factor': 2.0, 'name': string(node.layer_name + '_pow')}
            node.fluid_code.add_layer(
                "pow", inputs=input, output=node, param_attr=attr)
            attr = {
                'dim': dim[axis:],
                'keep_dim': False,
                'name': string(node.layer_name)
            }
            node.fluid_code.add_layer(
                "reduce_sum", inputs=node, output=node, param_attr=attr)
        else:  ## operation = MEAN
            attr = {
                'dim': dim[axis:],
                'keep_dim': False,
                'name': string(node.layer_name)
            }
            node.fluid_code.add_layer(
                "reduce_mean", inputs=node, output=node, param_attr=attr)
        attr = {'scale': coeff}
        node.fluid_code.add_layer(
            "scale", inputs=node, output=node, param_attr=attr)

    def deal_custom_layer(self, node):
        op = node.layer_type
        custom_code, func = make_custom_layer(node)
        params = get_params(node.layer, node.layer_type)
        arg_names, kwargs = set_args(func, params)
        kwargs['name'] = string(node.layer_name)
        kwargs['input_shape'] = node.input_shape
        data = node.data
        if data is not None:
            data = self.adjust_parameters(node)
            weights_name = deal_weights(node)
            for i in range(len(data)):
                self.weights[weights_name[i]] = data[i]
        inputs_node = []
        for i in range(len(node.inputs)):
            input = self.graph.get_bottom_node(node, idx=i, copy=True)
            if i == 1 and op == 'DetectionOutput':
                input = self.graph.get_bottom_node(node, idx=i, copy=True)
                while input is not None \
                      and input.layer_type != 'Softmax' \
                      and input.layer_type != 'Sigmoid':
                    input = self.graph.get_bottom_node(input, idx=0, copy=True)
                assert input is not None, 'This kind of DetectionOutput is not supported!'
                input = self.graph.get_bottom_node(input, idx=0, copy=True)
            inputs_node.append(input)
        node.fluid_code.add_layer(
            func.__code__.co_name,
            inputs=inputs_node,
            output=node,
            param_attr=kwargs,
            is_custom_layer=True)
        if op not in self.used_custom_layers:
            self.used_custom_layers[op] = custom_code

    def directly_map(self, node):
        assert node.layer_type in self.directly_map_ops
        op_info = self.directly_map_ops[node.layer_type]
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        attr = {'name': string(node.layer_name)}
        node.fluid_code.add_layer(
            op_info, inputs=input, output=node, param_attr=attr)
