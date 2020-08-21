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

import os
import sys
from google.protobuf import text_format
import numpy as np
from x2paddle.core.graph import GraphNode, Graph
from x2paddle.core.fluid_code import FluidCode
from x2paddle.op_mapper import caffe_shape


class CaffeResolver(object):
    def __init__(self, caffe_proto):
        self.caffe_proto = caffe_proto
        self.import_caffe()

    def import_caffepb(self):
        if self.caffe_proto is None:
            from x2paddle.decoder import caffe_pb2
            out = caffe_pb2
        else:
            if not os.path.isfile(self.caffe_proto):
                raise Exception(
                    "The .py file compiled by caffe.proto is not exist.")
            (filepath,
             tempfilename) = os.path.split(os.path.abspath(self.caffe_proto))
            (filename, extension) = os.path.splitext(tempfilename)
            sys.path.append(filepath)
            out = __import__(filename)
        return out

    def import_caffe(self):
        self.caffepb = self.import_caffepb()
        self.NetParameter = self.caffepb.NetParameter


class CaffeGraphNode(GraphNode):
    def __init__(self, layer, type_str, layer_name=None):
        if layer_name is None:
            super(CaffeGraphNode, self).__init__(
                layer, layer.name.replace('/', '_').replace('-', '_'))
        else:
            super(CaffeGraphNode, self).__init__(
                layer, layer_name.replace('/', '_').replace('-', '_'))
        self.layer_type = type_str
        self.fluid_code = FluidCode()
        self.data = None

    def set_params(self, params):
        self.data = params


class CaffeGraph(Graph):
    def __init__(self, model, params, caffe_pb):
        self.params = params
        self.caffe_pb = caffe_pb
        super(CaffeGraph, self).__init__(model)

    def filter_layers(self, layers):
        '''Filter out layers based on the current phase.'''
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        for layer in layers:
            if hasattr(layer, 'input'):
                continue
            type_str = self.get_layer_type(layer)
            phase = 'test'
            if len(layer.include):
                phase = phase_map[layer.include[0].phase]
            if len(layer.exclude):
                phase = phase_map[1 - layer.include[0].phase]
            exclude = (phase != 'test')
            # Dropout layers appear in a fair number of Caffe
            # test-time networks. These are just ignored. We'll
            # filter them out here.
            if (not exclude) and (phase == 'test'):
                exclude = (type_str == 'Dropout')
                if layer.type == 'Dropout':
                    drop_layer_top = layer.top[0]
                    drop_layer_bottom = layer.bottom[0]
                    if drop_layer_top != drop_layer_bottom:
                        for next_layer in layers:
                            for next_layer_bottom_idx, next_layer_bottom in enumerate(
                                    next_layer.bottom):
                                if drop_layer_top == next_layer_bottom:
                                    next_layer.bottom.remove(drop_layer_top)
                                    next_layer.bottom.insert(
                                        next_layer_bottom_idx,
                                        drop_layer_bottom)

            if not exclude:
                filtered_layers.append(layer)
                # Guard against dupes.
                assert layer.name not in filtered_layer_names
                filtered_layer_names.add(layer.name)
            else:
                print('The filter layer:' + layer.name)
        return filtered_layers

    def generate_input_layer(self, dims, index):
        dim_str = ''
        for dim in dims:
            dim_str += 'dim: {}\n'.format(str(dim))
        input_str = 'layer {\n'
        input_str += 'name: \"{}\"\n '.format(str(self.model.input[index]))
        input_str += 'type: "Input"\n'
        input_str += 'top: \"{}\"\n'.format(str(self.model.input[index]))
        input_str += 'input_param {\n'
        input_str += 'shape {\n'
        input_str += dim_str
        input_str += '}}}'
        input_str = str.encode(input_str)
        net = self.caffe_pb.NetParameter()
        text_format.Merge(input_str, net)
        return net.layers or net.layer

    def input2layers(self, input_layers=[]):
        inputs_num = len(self.model.input)
        if inputs_num != 0:
            input_dims_num = len(self.model.input_dim)
            if input_dims_num != 0:
                if input_dims_num > 0 and input_dims_num != inputs_num * 4:
                    raise Error('invalid input_dim[%d] param in prototxt' %
                                (input_dims_num))
                for i in range(inputs_num):
                    dims = self.model.input_dim[i * 4:(i + 1) * 4]
                    l = self.generate_input_layer(dims, i)
                    input_layers.append(l[0])
            else:
                for i in range(inputs_num):
                    dims = self.model.input_shape[i].dim[0:4]
                    l = self.generate_input_layer(dims, i)
                    input_layers.append(l[0])

    def transform_input_layers(self, layers, input_layers=[]):
        for layer in layers:
            if hasattr(layer, 'input'):
                input_dims_num = len(layers.input_dim)
                if input_dims_num > 0 and input_dims_num != 4:
                    raise Error('invalid input_dim[%d] param in prototxt' %
                                (input_dims_num))
                dims = self.model.input_dim[0:4]
                l = self.generate_input_layer(dims, i)
                input_layers.append(l[0])

    def get_layer_type(self, layer):
        if isinstance(layer.type, int):
            enum_values = self.caffe_pb._V1LAYERPARAMETER_LAYERTYPE.values
            vals = [val for val in enum_values if val.number == layer.type]
            part = vals[0].name.split('_')
            part = [s.capitalize() for s in part]
            type_str = ''
            type_str = type_str.join(part)
            if 'relu' in type_str.lower():
                type_str = type_str.replace('elu', 'eLU')
            elif type_str.lower() == 'lrn':
                type_str = 'LRN'
            return type_str
        else:
            return layer.type

    def build(self):
        layers = self.model.layers or self.model.layer

        layers = self.filter_layers(layers)

        input_layers = []

        self.input2layers(input_layers)
        self.transform_input_layers(layers, input_layers)
        layers = input_layers + layers
        for layer in layers:
            if hasattr(layer, 'name'):
                name = getattr(layer, 'name')
                setattr(layer, 'name', name.replace('/', '_').replace('-', '_'))
            for i, name in enumerate(layer.bottom):
                layer.bottom[i] = name.replace('/', '_').replace('-', '_')
            for i, name in enumerate(layer.top):
                layer.top[i] = name.replace('/', '_').replace('-', '_')

        top_layer = {}
        for layer in layers:
            if hasattr(layer, 'input'):
                continue
            type_str = self.get_layer_type(layer)
            self.node_map[layer.name] = CaffeGraphNode(layer, type_str)
            for in_name in layer.bottom:
                if in_name in top_layer:
                    self.connect(top_layer[in_name][-1], layer.name)
                else:
                    raise Exception(
                        'input[{}] of node[{}] does not exist in node_map'.
                        format(in_name, layer.name))
            for out_name in layer.top:
                if out_name not in top_layer:
                    top_layer[out_name] = [layer.name]
                else:
                    top_layer[out_name].append(layer.name)
        for layer_name, data in self.params:
            if layer_name in self.node_map:
                node = self.node_map[layer_name]
                node.set_params(data)
            else:
                print('Ignoring parameters for non-existent layer: %s' % \
                       layer_name)

        super(CaffeGraph, self).build()

    def get_bottom_node(self, node, idx=0, copy=False):
        input_node_name = node.inputs[idx]
        assert input_node_name in self.node_map, 'The {} isn\'t a valid node'.format(
            name)
        input_node = self.node_map[input_node_name]
        if len(input_node.layer.top) > 1:
            need_idx = list(input_node.layer.top).index(node.layer.bottom[idx])
            name = input_node_name + ':' + str(need_idx)
        else:
            name = input_node_name
        return self.get_node(name, copy=copy)


class CaffeDecoder(object):
    def __init__(self, proto_path, model_path, caffe_proto):
        self.proto_path = proto_path
        self.model_path = model_path

        self.resolver = CaffeResolver(caffe_proto=caffe_proto)
        self.net = self.resolver.NetParameter()
        with open(proto_path, 'rb') as proto_file:
            proto_str = proto_file.read()
            text_format.Merge(proto_str, self.net)

        self.load_using_pb()

        self.caffe_graph = CaffeGraph(self.net, self.params,
                                      self.resolver.caffepb)
        self.caffe_graph.build()

    def load_using_pb(self):
        data = self.resolver.NetParameter()
        data.MergeFromString(open(self.model_path, 'rb').read())
        layers = data.layers or data.layer
        for layer in layers:
            setattr(layer, 'name',
                    layer.name.replace('/', '_').replace('-', '_'))
        pair = lambda layer: (layer.name, self.normalize_pb_data(layer))
        self.params = [pair(layer) for layer in layers if layer.blobs]

    def normalize_pb_data(self, layer):
        transformed = []
        for blob in layer.blobs:
            if len(blob.shape.dim):
                dims = blob.shape.dim
                if layer.type == 'PReLU':
                    c_o, c_i, h, w = map(int, [1] + \
                        list(dims) + [1]* (3 - len(dims)))
                elif layer.type == 'Normalize' and len(dims) == 4:
                    data = np.asarray(list(blob.data), dtype=np.float32)
                    transformed.append(data)
                    continue
                else:
                    c_o, c_i, h, w = map(int,
                                         [1] * (4 - len(dims)) + list(dims))
            else:
                c_o = blob.num
                c_i = blob.channels
                h = blob.height
                w = blob.width
            data = np.asarray(
                list(blob.data), dtype=np.float32).reshape(c_o, c_i, h, w)

            transformed.append(data)
        return transformed
