# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from .opset_legacy import OpSet
import sys


def print_mapping_info(func):
    def run_mapping(*args, **kwargs):
        node = args[1]
        try:
            res = func(*args, **kwargs)
        except:
            raise Exception("convert failed node:{}, op_type is {}".format(
                node.name[9:], node.layer_type))
        else:
            return res

    return run_mapping


class OpSet7(OpSet):
    def __init__(self, decoder, paddle_graph):
        super(OpSet7, self).__init__(decoder, paddle_graph)
        self.directly_map_ops = {
            'Ceil': ['paddle.ceil'],
            # reduce function
            'ReduceMean': [
                'paddle.mean', dict(
                    axes='axis', keepdims='keepdim'), dict(
                        axes=None, keepdims=True)
            ],
            'ReduceMin': [
                'paddle.min', dict(
                    axes='axis', keepdims='keepdim'), dict(
                        axes=None, keepdim=True)
            ],
            'ReduceMax': [
                'paddle.max', dict(
                    axes='axis', keepdims='keepdim'), dict(
                        axes=None, keepdim=True)
            ],
            'ReduceProd': [
                'paddle.prod', dict(
                    axes='axis', keepdims='keepdim'), dict(
                        axes=None, keepdim=True)
            ],
            # active function
            'Relu': ['paddle.nn.ReLU'],
            'LeakyRelu': [
                'paddle.nn.LeakyReLU', dict(alpha='negative_slope'),
                dict(negative_slope=.01)
            ],
            'Elu':
            ['paddle.nn.functional.elu', dict(alpha='alpha'), dict(alpha=1.)],
            'ThresholdedRelu': [
                'paddle.nn.functional.thresholded_relu',
                dict(alpha='threshold'), dict(alpha=1.)
            ],
            'Tanh': ['paddle.nn.Tanh'],
            'Sigmoid': ['paddle.nn.Sigmoid'],
            'Softsign': ['paddle.nn.Softsign'],
            'Softplus': [
                'paddle.nn.Softplus', dict(threshold='threshold'),
                dict(threshold=float(sys.maxsize))
            ],
            'Exp': ['paddle.exp'],
            'Log': ['paddle.log'],
            'LogSoftmax': [
                'paddle.nn.functional.log_softmax', dict(axis='axis'),
                dict(axis=1)
            ],
            'Softmax': ['paddle.nn.Softmax', dict(axis='axis'), dict(axis=1)],
            'Sqrt': ['paddle.sqrt'],
            'Floor': ['paddle.floor'],
            'Abs': ['paddle.abs'],
            'Erf': ['paddle.erf'],
            'Sin': ['paddle.sin'],
            'Cos': ['paddle.cos'],
            'Atan': ['paddle.atan'],
            'Acos': ['paddle.acos'],
            'Asin': ['paddle.asin'],
            'IsInf': ['paddle.isinf'],
            'IsNaN': ['paddle.isnan'],
            'Cosh': ['paddle.cosh'],
            'Acosh': ['paddle.acosh'],
            'Asinh': ['paddle.asinh'],
            'Tan': ['paddle.tan'],
        }

    @print_mapping_info
    def Unsqueeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        # deal with scalar(0D) tensor
        if len(val_x.out_shapes[0]) == 0 and len(axes) == 1 and axes[0] == 0:
            self.paddle_graph.add_layer(
                'paddle.reshape',
                inputs={"x": val_x.name},
                outputs=[node.name],
                shape=[1])
        else:
            self.paddle_graph.add_layer(
                'paddle.unsqueeze',
                inputs={"x": val_x.name},
                axis=axes,
                outputs=[node.name])
