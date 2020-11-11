#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import copy
import numpy as np
from x2paddle.optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class Dygraph_TransposeElimination(FuseBase):
    def __init__(self):
        super(Dygraph_TransposeElimination, self).__init__(graph_type="dygraph")
        self.direct_layers = [
            'paddle.nn.ReLU', 'paddle.nn.ReLU6', 'paddle.abs',
            'paddle.nn.Sigmoid', 'paddle.exp', 'paddle.rsqrt',
            'paddle.nn.Swish', 'paddle.nn.Tanh',
            'paddle.nn.Softplus', 'paddle.nn.LeakyReLU',
            'paddle.floor', 'paddle.erf', 'paddle.square'
        ]
        self.elementwise_layers = [
            'paddle.add', 'fluid.layers.elementwise_sub',
            'paddle.multiply', 'paddle.divide'
        ]
        self.reduce_layers = [
            'paddle.mean', 'paddle.all',
            'paddle.max', 'paddle.any',
            'paddle.sum', 'paddle.prod'
        ]

    def get_transpose_num(self, graph):
        count = 0
        for layer_id, layer in graph.layers.items():
            if layer.kernel == "paddle.transpose":
                count += 1
        return count
    
    def operate(self, graph):
        total_layer_num = len(graph.layers)
        scanned_layers = set()
        optimized_transpose_layers = list()
        optimized_reduce_layers = list()
        optimized_concat_layers = list()
        optimized_elementwise_layers = list()

        def strip_transpose(_graph):
            layers = copy.deepcopy(_graph.layers)
            for layer_id, layer in layers.items():
                if layer_id in scanned_layers:
                    continue
                scanned_layers.add(layer_id)
                percent = round(len(scanned_layers) / total_layer_num * 100, 2)
                print("\rOptimize Transpose Layers...{}%".format(
                    percent))

                if layer.kernel != "paddle.transpose":
                    continue
                if layer.attrs["perm"] != [0, 2, 3, 1]:
                    continue
                transpose_layers = list()
                propagate_layers = list()
                reduce_layers = list()
                concat_layers = list()
                # 此elementwise_layers专用于存储shape(4) + shape(1)的形式layer
                elementwise_layers = list()
                can_be_optimized = True
                for out in _graph.edges_out.get(layer_id, []):
                    if _graph.layers[out].kernel == "paddle.transpose":
                        if _graph.layers[out].attrs["perm"] != [0, 3, 1, 2]:
                            can_be_optimized = False
                            break
                        transpose_layers.append(out)
                    elif _graph.layers[out].kernel in self.elementwise_layers:
                        propagate_layers.append(out)
                    elif _graph.layers[out].kernel in self.direct_layers:
                        ouput_index = 1 if _graph.layers[out].kernel.startswith("paddle.nn.") else 0
                        if _graph.layers[out].outputs[ouput_index] in _graph.outputs:
                            can_be_optimized = False
                            break
                        propagate_layers.append(out)
                    elif _graph.layers[out].kernel in self.reduce_layers:
                        if _graph.layers[out].outputs[0] in _graph.outputs:
                            can_be_optimized = False
                            break
                        if _graph.layers[out].attrs.get('keepdim', False):
                            can_be_optimized = False
                            break
                        propagate_layers.append(out)
                        reduce_layers.append(out)
                    elif _graph.layers[out].kernel == "paddle.concat":
                        if _graph.layers[out].outputs[0] in _graph.outputs:
                            can_be_optimized = False
                            break
                        propagate_layers.append(out)
                        concat_layers.append(out)
                    else:
                        can_be_optimized = False
                        break

                visited_layers = set()
                while len(propagate_layers) > 0 and can_be_optimized:
                    current_id = propagate_layers.pop(0)
                    visited_layers.add(current_id)
                    for out in _graph.edges_out.get(current_id, []):
                        if _graph.layers[
                                out].kernel == "paddle.transpose":
                            if _graph.layers[out].attrs["perm"] != [0, 3, 1, 2]:
                                can_be_optimized = False
                                break
                            transpose_layers.append(out)
                        elif _graph.layers[
                                out].kernel in self.elementwise_layers:
                            if _graph.layers[out].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                propagate_layers.append(out)
                        elif _graph.layers[out].kernel in self.direct_layers:
                            ouput_index = 1 if _graph.layers[out].kernel.startswith("paddle.nn.") else 0
                            if _graph.layers[out].outputs[ouput_index] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                propagate_layers.append(out)
                        elif _graph.layers[out].kernel in self.reduce_layers:
                            if _graph.layers[out].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if _graph.layers[out].attrs.get('keepdim',
                                                                False):
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                propagate_layers.append(out)
                                reduce_layers.append(out)
                        elif _graph.layers[out].kernel == "paddle.concat":
                            if _graph.layers[out].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                propagate_layers.append(out)
                                concat_layers.append(out)
                        else:
                            can_be_optimized = False
                            break
                    for ipt in _graph.edges_in.get(current_id, []):
                        if _graph.layers[
                                current_id].kernel in self.elementwise_layers:
                            try:
                                x_shape = _graph.layers[
                                    current_id].input_shapes['x']
                                y_shape = _graph.layers[
                                    current_id].input_shapes['y']
                                if _graph.layers[ipt].outputs[
                                        0] == _graph.layers[current_id].inputs[
                                            'x']:
                                    if len(x_shape) <= 1:
                                        elementwise_layers.append(current_id)
                                        continue
                                elif _graph.layers[ipt].outputs[
                                        0] == _graph.layers[current_id].inputs[
                                            'y']:
                                    if len(y_shape) <= 1:
                                        elementwise_layers.append(current_id)
                                        continue
                                else:
                                    raise Exception(
                                        "Unexcepted situation happend while optimizing transpose"
                                    )
                            except Exception as e:
                                can_be_optimized = False
                                break
                        if _graph.layers[
                                ipt].kernel == "paddle.transpose":
                            if _graph.layers[ipt].attrs["perm"] != [0, 2, 3, 1]:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                transpose_layers.append(ipt)
                        elif _graph.layers[
                                ipt].kernel in self.elementwise_layers:
                            if _graph.layers[ipt].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                propagate_layers.append(ipt)
                        elif _graph.layers[ipt].kernel in self.direct_layers:
                            ouput_index = 1 if _graph.layers[ipt].kernel.startswith("paddle.nn.") else 0
                            if _graph.layers[ipt].outputs[ouput_index] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                propagate_layers.append(ipt)
                        elif _graph.layers[ipt].kernel in self.reduce_layers:
                            if _graph.layers[ipt].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if _graph.layers[ipt].attrs.get('keepdim',
                                                                False):
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                propagate_layers.append(ipt)
                                reduce_layers.append(ipt)
                        elif _graph.layers[ipt].kernel == "paddle.concat":
                            if _graph.layers[ipt].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                propagate_layers.append(ipt)
                                concat_layers.append(ipt)
                        else:
                            can_be_optimized = False
                            break
                    if not can_be_optimized:
                        break
                if not can_be_optimized:
                    continue

                transpose_layers.append(layer_id)
                transpose_layers = list(set(transpose_layers))
                for l in transpose_layers:
                    if graph.layers[l].outputs[0] in graph.outputs:
                        can_be_optimized = False
                        break
                if not can_be_optimized:
                    continue

                for l in transpose_layers:
                    self.delete_layer_with_associated(_graph, l)

                optimized_transpose_layers.extend(transpose_layers)
                optimized_reduce_layers.extend(reduce_layers)
                optimized_concat_layers.extend(concat_layers)
                optimized_elementwise_layers.extend(elementwise_layers)
                return True
            return False

        before_transpose_num = self.get_transpose_num(graph)
        opt_graph = copy.deepcopy(graph)
        total_layer_num = len(opt_graph.layers)

        while strip_transpose(opt_graph):
            pass

        for layer_id in list(set(optimized_transpose_layers)):
            self.delete_layer_with_associated(graph, layer_id)
        for layer_id in list(set(optimized_reduce_layers)):
            dim = graph.layers[layer_id].attrs.get('dim', None)
            if dim is not None:
                for i in range(len(dim)):
                    dim[i] = [0, 2, 3, 1][dim[i]]
                graph.layers[layer_id].attrs['dim'] = dim
        for layer_id in list(set(optimized_concat_layers)):
            axis = graph.layers[layer_id].attrs.get('axis', 0)
            graph.layers[layer_id].attrs['axis'] = [0, 2, 3, 1][axis]
        for layer_id in list(set(optimized_elementwise_layers)):
            axis = graph.layers[layer_id].attrs.get('axis', -1)
            graph.layers[layer_id].attrs['axis'] = [0, 2, 3, 1][axis]

        current_transpose_num = self.get_transpose_num(graph)
        print(
            "\nTranspose layers optimized, before: transpose_num={}, after: transpose_num={}".
            format(before_transpose_num, current_transpose_num))