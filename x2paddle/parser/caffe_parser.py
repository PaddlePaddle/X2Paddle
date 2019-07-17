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


class CaffeResolver(object):
    def __init__(self, use_default=True):
        self.use_default = use_default
        self.import_caffe()

    def import_caffepb(self):
        p = os.path.realpath(__file__)
        p = os.path.dirname(p)
        p = os.path.join(p, '../proto')
        sys.path.insert(0, p)
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
            super(CaffeGraphNode, self).__init__(layer, layer.name)
        else:
            super(CaffeGraphNode, self).__init__(layer, layer_name)
        self.layer_type = layer.type

    def set_params(self, params):
        self.data = params


class CaffeGraph(Graph):
    def __init__(self, resolver, model, params):
        self.params = params
        if resolver.has_pycaffe():
            self.did_use_pb = False
        else:
            self.did_use_pb = True
        super(CaffeGraph, self).__init__(model)

    def filter_layers(self, layers):
        '''Filter out layers based on the current phase.'''
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        print('The filter layer:')
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
                print(layer.name)
        return filtered_layers

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
        return data

    def build(self):
        layers = self.model.layers or self.model.layer
        layers = self.filter_layers(layers)

        inputs_num = len(self.model.input)
        if inputs_num != 0:
            input_dims_num = len(self.model.input_dim)
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
                    raise Error(
                        'You must install the caffe first when you use old style prototxt.'
                    )
                data.name = self.model.input[0]
                data.top[0] = self.model.input[0]

        for layer in layers:
            self.node_map[layer.name] = CaffeGraphNode(layer)

        for layer_name, node in self.node_map.items():
            for in_node in node.layer.bottom:
                if in_node in self.node_map:
                    self.connect(in_node, layer_name)
                else:
                    raise Exception(
                        'input[{}] of node[{}] does not exist in node_map'.
                        format(in_node, layer_name))

        for layer_name, data in self.params:
            if layer_name in self.node_map:
                node = self.node_map[layer_name]
                node.set_params(self.adjust_parameters(node, data))
            else:
                notice('Ignoring parameters for non-existent layer: %s' % \
                        layer_name)
        super(CaffeGraph, self).build()


class CaffeParser(object):
    def __init__(self, proto_path, model_path, use_caffe=True):
        self.proto_path = proto_path
        self.model_path = model_path

        self.resolver = CaffeResolver(use_default=use_caffe)
        self.net = self.resolver.NetParameter()
        with open(proto_path, 'rb') as proto_file:
            proto_str = proto_file.read()
            text_format.Merge(proto_str, self.net)

        self.load()
        self.caffe_graph = CaffeGraph(self.resolver, self.net, self.params)
        self.caffe_graph.build()

    def load(self):
        if self.resolver.has_pycaffe():
            self.load_using_caffe()
        else:
            self.load_using_pb()

    def load_using_caffe(self):
        caffe = self.resolver.caffe
        caffe.set_mode_cpu()
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
