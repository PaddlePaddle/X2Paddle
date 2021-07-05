#   Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import tflite


def _decode_type(n):
    _tflite_m = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
    }
    return _tflite_m[n]


class TFLiteDecoder(object):
    def __init__(self, tflite_path):
        with open(tflite_path, 'rb') as f:
            buf = f.read()
        buf = bytearray(buf)
        model = tflite.Model.GetRootAsModel(buf, 0)
        assert model.SubgraphsLength(
        ) == 1, "only support one subgraph (main subgraph)"
        self.graph = model.Subgraphs(0)
        self.model = model
        self.get_tensor_shape_dtype()

    def get_tensor_shape_dtype(self):
        self.shape_dict = dict()
        self.dtype_dict = dict()
        self.inputs_name = list()
        input_index = list()
        input_cnt = self.graph.InputsLength()
        for i in range(input_cnt):
            input_index.append(self.graph.Inputs(i))
        tensor_cnt = self.graph.TensorsLength()
        for i in range(tensor_cnt):
            tensor = self.graph.Tensors(i)
            shape = tuple(tensor.ShapeAsNumpy())
            tensor_type = tensor.Type()
            name = tensor.Name().decode("utf8")
            if i in input_index:
                self.inputs_name.append(name)
            self.shape_dict[name] = [int(x) for x in shape]
            self.dtype_dict[name] = _decode_type(tensor_type)
