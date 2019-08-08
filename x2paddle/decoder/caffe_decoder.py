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
    def __init__(self, layer, layer_name=None):
        if layer_name is None:
            super(CaffeGraphNode,
                  self).__init__(layer,
                                 layer.name.replace('/', '_').replace('-', '_'))
        else:
            super(CaffeGraphNode,
                  self).__init__(layer,
                                 layer_name.replace('/', '_').replace('-', '_'))
        self.layer_type = layer.type
        self.fluid_code = FluidCode()
        self.data = None

    def set_params(self, params):
        self.data = params


class CaffeGraph(Graph):
    def __init__(self, model, params):
        self.params = params
        super(CaffeGraph, self).__init__(model)

    def filter_layers(self, layers):
        '''Filter out layers based on the current phase.'''
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        for layer in layers:
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
                exclude = (layer.type == 'Dropout')
            if not exclude:
                filtered_layers.append(layer)
                # Guard against dupes.
                assert layer.name not in filtered_layer_names
                filtered_layer_names.add(layer.name)
            else:
                print('The filter layer:' + layer.name)
        return filtered_layers

    def build(self):
        layers = self.model.layers or self.model.layer
        layers = self.filter_layers(layers)

        top_layer = {}
        for layer in layers:
            self.node_map[layer.name] = CaffeGraphNode(layer)
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
            proto_str = self.old2new(proto_file)
            text_format.Merge(proto_str, self.net)

        self.load_using_pb()
        self.caffe_graph = CaffeGraph(self.net, self.params)
        self.caffe_graph.build()

    def load_using_pb(self):
        data = self.resolver.NetParameter()
        data.MergeFromString(open(self.model_path, 'rb').read())
        pair = lambda layer: (layer.name, self.normalize_pb_data(layer))
        layers = data.layers or data.layer
        self.params = [pair(layer) for layer in layers if layer.blobs]

    def normalize_pb_data(self, layer):
        transformed = []
        for blob in layer.blobs:
            if len(blob.shape.dim):
                dims = blob.shape.dim
                c_o, c_i, h, w = map(int, [1] * (4 - len(dims)) + list(dims))
            else:
                c_o = blob.num
                c_i = blob.channels
                h = blob.height
                w = blob.width
            data = np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w)
            transformed.append(data)
        return transformed

    def old2new(self, proto_file):
        part1_str = ''
        part2_str = ''
        part3_str = ''
        is_input = False
        dims = []
        line = proto_file.readline()
        print('Check if it is a new style of caffe...')
        while line:
            l_str = bytes.decode(line)
            if l_str.replace(' ', '').startswith('input:'):
                part2_str += 'layer {\n'
                part2_str += (
                    '  name: ' +
                    l_str.strip().replace(' ', '').split('input:')[-1] + '\n')
                part2_str += '  type: \"Input\"\n'
                part2_str += (
                    '  top: ' +
                    l_str.strip().replace(' ', '').split('input:')[-1] + '\n')
                is_input = True
                line = proto_file.readline()
                continue
            elif l_str.replace(' ', '').startswith('input_dim:'):
                dims.append(
                    int(l_str.strip().replace(' ', '').split('input_dim:')[-1]))
                if len(dims) == 4:
                    part2_str += '  input_param { shape: { dim: ' + str(dims[0]) + \
                                               ' dim: ' + str(dims[1]) + \
                                               ' dim: ' + str(dims[2]) + \
                                               ' dim: ' + str(dims[3]) + ' } }\n'
                    dims = []
                    part2_str += '}\n'
                line = proto_file.readline()
                if bytes.decode(line).replace(' ', '').startswith('}'):
                    line = proto_file.readline()
                continue
            elif l_str.replace(' ', '').startswith('input_shape'):
                part2_str += l_str.replace('input_shape',
                                           'input_param { shape: ')
                l_str = bytes.decode(proto_file.readline())
                while l_str:
                    if '}' in l_str:
                        part2_str += l_str + '\n}\n}'
                        break
                    else:
                        part2_str += l_str
                    l_str = bytes.decode(proto_file.readline())
                line = proto_file.readline()
                continue
            if not is_input:
                part1_str += bytes.decode(line)
            else:
                part3_str += bytes.decode(line)
            line = proto_file.readline()
        out = part1_str + part2_str + part3_str
        layer_str = 'layer{'
        part = out.split(layer_str)
        if len(part) == 1:
            layer_str = 'layer {'
            part = out.split(layer_str)
        for i in range(len(part)):
            if part[i].strip().replace(' ', '') == '' or part[i].count(':') > 1:
                continue
            out = out.replace(layer_str + part[i], part[i].replace(' ', ''))
        return str.encode(out)
