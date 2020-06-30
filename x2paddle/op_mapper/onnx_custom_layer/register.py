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
""" this module provides 'register' for registering customized layers
"""

g_custom_layers = {}


def register(kind, shape, layer, child_func, weights):
    """ register a custom layer or a list of custom layers

    Args:
        @kind (str or list): type name of the layer
        @shape (function): a function to generate the shape of layer's output
        @layer (function): a function to generate the paddle code of layer
        @weights (function): a function to deal with weights data

    Returns:
        None
    """
    assert type(shape).__name__ == 'function', 'shape should be a function'
    assert type(layer).__name__ == 'function', 'layer should be a function'

    if type(kind) is str:
        kind = [kind]
    else:
        assert type(
            kind) is list, 'invalid param "kind" for register, not a list or str'

    for k in kind:
        assert type(
            k) is str, 'invalid param "kind" for register, not a list of str'
        assert k not in g_custom_layers, 'this type[%s] has already been registered' % (
            k)
        g_custom_layers[k] = {
            'shape': shape,
            'layer': layer,
            'child_func': child_func,
            'weights': weights
        }


def get_registered_layers():
    return g_custom_layers
