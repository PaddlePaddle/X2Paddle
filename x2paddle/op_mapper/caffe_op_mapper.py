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
from x2paddle.op_mapper.caffe_custom_layer import *


class CaffeOpMapper(OpMapper):
    def __init__(self, decoder):
        super(CaffeOpMapper, self).__init__()
        self.graph = decoder.caffe_graph
        self.weights = dict()
        resolver = decoder.resolver
        self.used_custom_layers = {}
        if resolver.has_pycaffe():
            self.did_use_pb = False
        else:
            self.did_use_pb = True

        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                self.set_shape(node)
                func = getattr(self, op)
                func(node)
            elif op in custom_layers:
                self.set_shape(node, is_fluid_op=False)
                self.deal_custom_layer(node)
            else:
                raise Exception("Model are not supported yet.")

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

    def set_shape(self, node, is_fluid_op=True):
        inputs = node.inputs
        input_shape = []
        for i, nm in enumerate(inputs):
            last_node = self.graph.get_node(nm)
            tmp = node.layer.bottom[i]
            idx = list(last_node.layer.top).index(tmp)
            input_shape.append(last_node.output_shape[idx])
        node.set_input_shape(input_shape)
        if is_fluid_op:
            node.set_output_shape(input_shape)
        else:
            node.set_output_shape(compute_output_shape(node),
                                  is_input=is_fluid_op)

    def adjust_parameters(self, node):
        data = node.data
        if not self.did_use_pb:
            return data
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
            if len(shape_old) != shape_new:
                print('squeeze idx:%d, with kind:%s,name:%s' % \
                        (idx, node.layer_type, node.layer.name))
        return data

    def get_kernel_parameters(self, kind, params):
        assert kind in ['Convolution', 'Pooling', 'Deconvolution']
        [k_h, k_w] = [1, 1]
        if isinstance(params.kernel_size, numbers.Number):
            [k_h, k_w] = [params.kernel_size] * 2
        elif len(params.kernel_size) > 0:
            k_h = params.kernel_h if params.kernel_h else params.kernel_size[0]
            k_w = params.kernel_w if params.kernel_w else params.kernel_size[
                len(params.kernel_size) - 1]
        [s_h, s_w] = [1, 1]
        if isinstance(params.stride, numbers.Number):
            [s_h, s_w] = [params.stride] * 2
        elif len(params.stride) > 0:
            s_h = params.stride_h if params.stride_h else params.stride[0]
            s_w = params.stride_w if params.stride_w else params.stride[
                len(params.stride) - 1]
        [p_h, p_w] = [0, 0]
        if isinstance(params.pad, numbers.Number):
            [p_h, p_w] = [params.pad] * 2
        elif len(params.pad) > 0:
            p_h = params.pad_h if params.pad_h else params.pad[0]
            p_w = params.pad_w if params.pad_w else params.pad[len(params.pad) -
                                                               1]
        dila_h = dila_w = 1
        group = 1
        c_o = 1
        if kind in ['Convolution', 'Deconvolution', 'ConvolutionDepthwise']:
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
        data = self.adjust_parameters(node)
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
        data = self.adjust_parameters(node)
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
        attr = {'axis': axis, 'name': string(node.layer_name + '_softmax')}
        node.fluid_code.add_layer("softmax",
                                  inputs=input,
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
                'ends': [points[i + 1]]
            }
            node.fluid_code.add_layer("slice",
                                      inputs=input,
                                      output=node.layer_name + '_' + str(i),
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
            'mode': string(mode),
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
            node.inputs) == 1, 'The count of Scale node\'s input is not 1.'
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
            if self.is_Scale(input):
                tmp = self.graph.get_bottom_node(input, idx=0, copy=True)
                if self.is_BN(tmp):
                    input = tmp
            inputs_node.append(input)
        node.fluid_code.add_layer(func.__code__.co_name,
                                  inputs=inputs_node,
                                  output=node,
                                  param_attr=kwargs,
                                  is_custom_layer=True)
        if op not in self.used_custom_layers:
            self.used_custom_layers[op] = custom_code
