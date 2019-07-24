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
            'ceil_mode': True,
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

    def Slice(self, node):
        assert len(
            node.inputs) == 1, 'The count of Slice node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
        attr = {'name': string(node.layer_name)}
        node.fluid_code.add_layer("sigmoid",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def AbsVal(self, node):
        assert len(
            node.inputs) == 1, 'The count of PReLU node\'s input is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
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
                inputs[1] = self.graph.get_bottom_node(node, idx=i, copy=True)
            else:
                inputs[0] = self.graph.get_bottom_node(node, idx=i, copy=True)
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
        inputs.append(self.graph.get_bottom_node(node, idx=0, copy=True))
        inputs.append(self.graph.get_bottom_node(node, idx=1, copy=True))
        if mode == 0:
            attr = {'act': None, 'name': string(node.layer_name)}
            node.fluid_code.add_layer("elementwise_mul",
                                      inputs=inputs,
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
                attr = {'act': None, 'name': string(node.layer_name)}
                node.fluid_code.add_layer("elementwise_add",
                                          inputs=inputs,
                                          output=node,
                                          param_attr=attr)
        else:
            attr = {'act': None, 'name': string(node.layer_name)}
            node.fluid_code.add_layer("elementwise_max",
                                      inputs=inputs,
                                      output=node,
                                      param_attr=attr)

    def BatchNorm(self, node):
        assert len(node.inputs) == 1 and len(
            node.outputs
        ) == 1, 'The count of BatchNorm node\'s input and output is not 1.'
        input = self.graph.get_bottom_node(node, idx=0, copy=True)
        params = node.layer.batch_norm_param
        if hasattr(params, eps):
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
            params = node.layer.scale_params
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
                input1 = self.graph.get_bottom_node(node, idx=1, copy=True)
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
