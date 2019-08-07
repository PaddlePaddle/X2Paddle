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

from collections import OrderedDict as _dict

default_op_mapping_field_values = _dict()
default_op_mapping_field_values['FLUID_OP'] = ''
default_op_mapping_field_values['FLUID_INPUT_ARGS'] = None
default_op_mapping_field_values['FLUID_OUTPUT_ARGS'] = None
default_op_mapping_field_values['ATTR_MAPPING'] = dict()  # dict(onnx_attr_from=fluid_attr_to)
default_op_mapping_field_values['DEFAULTS'] = dict()  # dict(fluid_attr=default)
default_op_mapping_field_values['INPUT_PERM'] = None  # sampler: [idx_onnx_arg...]
default_op_mapping_field_values['OUTPUT_PERM'] = None  # sampler: [idx_onnx_arg...]
default_op_mapping_field_values['FILL_NAME_FIELD'] = True
default_op_mapping = {
#         ## unary ops ##
#         'Abs': ['abs', ['X'], ['Out']],
#         'ArgMax': ['argmax', ['X'], ['Out'], dict(keepdims='')],
#         'ArgMin': ['argmin', ['X'], ['Out'], dict(keepdims='')],
#         'Ceil': ['ceil', ['X'], ['Out']],
#         'Clip': ['clip', ['X'], ['Out']], # attrs bypassed
#         'Cos': ['cos', ['X'], ['Out']],
#         'Elu': ['elu', ['X'], ['Out']],
#         'Exp': ['exp', ['X'], ['Out']],
#         'Flatten': ['flatten', ['X'], ['Out']], # attrs bypassed, FIXME: emit flatten2
#         'Floor': ['floor', ['X'], ['Out']],
        'Gather': ['gather', ['X'], ['Out'], dict(axis='')],
#         'LeakyRelu': ['leaky_relu', ['X'], ['Out']],
#         'Log': ['log', ['X'], ['Out']],
#         'Relu': ['relu', ['X'], ['Out']],
#         'Selu': ['selu', ['X'], ['Out'], dict(gamma='scale')],
        'Shape': ['shape', ['X'], ['Out']], # FIXME: out is int64 vs int32
#         'Sigmoid': ['sigmoid', ['X'], ['Out']],
#         'Sin': ['sin', ['X'], ['Out']],
#         'Squeeze': ['squeeze', ['X'], ['Out']], # attrs bypassed, FIXME: emit squeeze2
#         'Softplus': ['softplus', ['X'], ['Out']],
#         # FIXME: default axis = -1, reshape required before and after
#         'Softmax': ['softmax', ['X'], ['Out'], dict(axis='')],
#         'Softsign': ['softsign', ['X'], ['Out']],
#         'Sqrt': ['sqrt', ['X'], ['Out']],
#         'Tanh': ['tanh', ['X'], ['Out']],
#         'ThresholdedRelu': ['thresholded_relu', ['X'], ['Out'], dict(alpha='threshold')],
#         ## binary ops ##
#         'Add': ['elementwise_add', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
#         'And': ['logical_and', ['X', 'Y'], ['Out']],
#         'Div': ['elementwise_div', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
#         'Equal': ['equal', ['X', 'Y'], ['Out'], dict(), dict(), None, None, False],
#         'Greater': ['less_than', ['X', 'Y'], ['Out'], dict(), dict(), [1, 0], None, False],
#         'Less': ['less_than', ['X', 'Y'], ['Out'], dict(), dict(), None, None, False],
#         'MatMul': ['matmul', ['X', 'Y'], ['Out']], # defaults excluded for transpose_x vs transpose_X
#         'Max': ['elementwise_max', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
#         'Min': ['elementwise_min', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Mul': ['elementwise_mul', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
#         'Not': ['logical_not', ['X', 'Y'], ['Out']],
#         'Pow': ['elementwise_pow', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)], # TODO: pow for scalar exponent
#         'Sub': ['elementwise_sub', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
#         # reduce ops
#         'ReduceMax': ['reduce_max', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
#         'ReduceMean': ['reduce_mean', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
#         'ReduceMin': ['reduce_min', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
#         'ReduceProd': ['reduce_prod', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
#         'ReduceSum': ['reduce_sum', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
#         # other ops
#         'Scatter': ['scatter', ['X', 'Index', 'Updates'], ['Out']],
#         'TopK': ['topk', ['X', 'K'], ['Out', 'Indices']],
#         'Expand': ['expand', ['X'], ['Out'], dict(shape='expand_times')],
}

default_ioa_constraint= {
#     'ArgMax': [
#         (lambda i, o, a: a.get('keepdims', 1) == 1,
#          'only keepdims = 0 is supported')
#     ],
#     'ArgMin': [
#         (lambda i, o, a: a.get('keepdims', 1) == 1,
#          'only keepdims = 0 is supported')
#     ],
    'Gather': [
        (lambda i, o, a: a.get('axis', 0) == 0, 'only axis = 0 is supported')
    ],
#     'Shrink': [
#         (lambda i, o, a: a.get('bias', 0) == a.get('lambd', 0.5),
#          'only SoftShrink with bias = lambd is supported')
#     ],
#     'OneHot': [
#         (lambda i, o, a: a.get('axis', -1) == -1,
#          'only axis = -1 is supported')
#     ],
#     'Scatter': [
#         (lambda i, o, a: a.get('axis', 0) == 0, 'only axis = 0 is supported')
#     ],
#     'TopK': [
#         (lambda i, o, a: a.get('axis', -1) == -1,
#          'only axis = -1 is supported'),
#     ]
}
