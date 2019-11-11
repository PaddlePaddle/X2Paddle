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
import numpy as _np

default_op_mapping_field_values = _dict()
default_op_mapping_field_values['FLUID_OP'] = ''
default_op_mapping_field_values['FLUID_INPUT_ARGS'] = None
default_op_mapping_field_values['FLUID_OUTPUT_ARGS'] = None
default_op_mapping_field_values['ATTR_MAPPING'] = dict()
default_op_mapping_field_values['DEFAULTS'] = dict()
default_op_mapping_field_values['INPUT_PERM'] = None
default_op_mapping_field_values['OUTPUT_PERM'] = None
default_op_mapping_field_values['FILL_NAME_FIELD'] = True

default_op_mapping = {
    'Shape': ['shape', ['X'], ['Out']],
    'Clip': [
        'clip', ['X'], ['Out'],
        dict(),
        dict(
            min=(_np.asarray([255, 255, 127, 255],
                             dtype=_np.uint8).view(_np.float32)),
            max=(_np.asarray([255, 255, 127, 127],
                             dtype=_np.uint8).view(_np.float32)),
        )
    ],
    'Ceil': ['ceil', ['X'], ['Out']],
    'ReduceMean': [
        'reduce_mean', ['X'], ['Out'],
        dict(axes='dim', keepdims='keep_dim'),
        dict(keep_dim=1)
    ],
    'ReduceSum': [
        'reduce_sum', ['X'], ['Out'],
        dict(axes='dim', keepdims='keep_dim'),
        dict(keep_dim=1)
    ],
    'ReduceMin': [
        'reduce_min', ['X'], ['Out'],
        dict(axes='dim', keepdims='keep_dim'),
        dict(keep_dim=1)
    ],
    #active function
    'Relu': ['relu', ['X'], ['Out']],
    'LeakyRelu': ['leaky_relu', ['X'], ['Out'],
                  dict(), dict(alpha=.01)],
    'Elu': ['elu', ['X'], ['Out'],
            dict(), dict(alpha=1.)],
    'ThresholdedRelu': [
        'thresholded_relu', ['X'], ['Out'],
        dict(alpha='threshold'),
        dict(alpha=1.)
    ],
    'Tanh': ['tanh', ['X'], ['Out']],
    'Sigmoid': ['sigmoid', ['X'], ['Out']],
    'HardSigmoid': [
        'hard_sigmoid', ['X'], ['Out'],
        dict(alpha='slope', beta='offset'),
        dict(slope=.2, offset=.5)
    ],
    'Softsign': ['softsign', ['X'], ['Out']],
    'Softplus': ['softplus', ['X'], ['Out']],
    'Exp': ['exp', ['X'], ['Out']],
    'Softmax': ['softmax', ['X'], ['Out'],
                dict(), dict(axis=1)],
    'Sqrt': ['sqrt', ['X'], ['Out']],
    'Floor': ['floor', ['X'], ['Out']],
    'Abs': ['abs', ['X'], ['Out']],
}

default_ioa_constraint = {
    'Gather':
    [(lambda i, o, a: a.get('axis', 0) == 0, 'only axis = 0 is supported')],
}
