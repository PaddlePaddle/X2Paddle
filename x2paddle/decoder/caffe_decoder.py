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
    def __init__(self, caffe_proto_folder=None):
        self.proto_path = caffe_proto_folder
        if self.proto_path == None:
            self.use_default = True
        else:
            self.use_default = False
        self.import_caffe()

    def import_caffepb(self):
        sys.path.append(self.proto_path)
        import caffe_pb2
        return caffe_pb2

    def import_caffe(self):
        self.caffe = None
        self.caffepb = None
        if self.use_default:
            try:
                # Try to import PyCaffe first
                import caffe
                self.caffe = caffe
            except ImportError:
                # Fall back to the protobuf implementation
                self.caffepb = self.import_caffepb()
        else:
            self.caffepb = self.import_caffepb()
        if self.caffe:
            # Use the protobuf code from the imported distribution.
            # This way, Caffe variants with custom layers will work.
            self.caffepb = self.caffe.proto.caffe_pb2
        self.NetParameter = self.caffepb.NetParameter

    def has_pycaffe(self):
        return self.caffe is not None


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

    def set_output_shape(self, input_shape, is_input=True):
        func_name = 'shape_' + self.layer_type.lower()
        if is_input:
            self.output_shape = getattr(caffe_shape, func_name)(self.layer,
                                                                input_shape)
        else:
            self.output_shape = input_shape

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape


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

        inputs_num = len(self.model.input)
        if inputs_num != 0:
            input_dims_num = len(self.model.input_dim)
            if input_dims_num != 0:
                if input_dims_num > 0 and input_dims_num != inputs_num * 4:
                    raise Error('invalid input_dim[%d] param in prototxt' %
                                (input_dims_num))
                for i in range(inputs_num):
                    dims = self.model.input_dim[i * 4:(i + 1) * 4]
                    data = self.model.layer.add()
                    try:
                        from caffe import layers as L
                        data.CopyFrom(
                            L.Input(input_param=dict(shape=dict(
                                dim=[dims[0], dims[1], dims[2], dims[3]
                                     ]))).to_proto().layer[0])
                    except:
                        raise ImportError(
                            'The .proto file does not work for the old style prototxt. You must install the caffe or modify the old style to new style in .protottx file.'
                        )
                    data.name = self.model.input[i]
                    data.top[0] = self.model.input[i]
            else:
                for i in range(inputs_num):
                    dims = self.model.input_shape[i].dim[0:4]
                    data = self.model.layer.add()
                    try:
                        from caffe import layers as L
                        data.CopyFrom(
                            L.Input(input_param=dict(shape=dict(
                                dim=[dims[0], dims[1], dims[2], dims[3]
                                     ]))).to_proto().layer[0])
                    except:
                        raise ImportError(
                            'The .proto file does not work for the old style prototxt. You must install the caffe or modify the old style to new style in .protottx file.'
                        )
                    data.name = self.model.input[i]
                    data.top[0] = self.model.input[i]
            layers = [data] + layers

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
                raise Exception('Ignoring parameters for non-existent layer: %s' % \
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
    def __init__(self, proto_path, model_path, caffe_proto_folder=None):
        self.proto_path = proto_path
        self.model_path = model_path

        self.resolver = CaffeResolver(caffe_proto_folder=caffe_proto_folder)
        self.net = self.resolver.NetParameter()
        with open(proto_path, 'rb') as proto_file:
            proto_str = proto_file.read()
            text_format.Merge(proto_str, self.net)

        self.load()
        self.caffe_graph = CaffeGraph(self.net, self.params)
        self.caffe_graph.build()

    def load(self):
        if self.resolver.has_pycaffe():
            self.load_using_caffe()
        else:
            self.load_using_pb()

    def load_using_caffe(self):
        caffe = self.resolver.caffe
        caffe.set_mode_cpu()
        print(self.proto_path)
        print(self.model_path)
        net = caffe.Net(self.proto_path, self.model_path, caffe.TEST)
        data = lambda blob: blob.data
        self.params = [(k, list(map(data, v))) for k, v in net.params.items()]

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
