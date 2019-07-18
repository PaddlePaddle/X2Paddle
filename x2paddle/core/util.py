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

from paddle.fluid.proto import framework_pb2
import struct
import math
import os


def string(param):
    return "\'{}\'".format(param)


def get_same_padding(in_size, kernel_size, stride):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0

    return [pad0, pad1]


def export_paddle_param(param, param_name, dir):
    dtype_map = {
        "int16": [framework_pb2.VarType.INT16, 'h'],
        "int32": [framework_pb2.VarType.INT32, 'i'],
        "int64": [framework_pb2.VarType.INT64, 'q'],
        "float16": [framework_pb2.VarType.FP16, 'e'],
        "float32": [framework_pb2.VarType.FP32, 'f'],
        "float64": [framework_pb2.VarType.FP64, 'd']
    }
    shape = param.shape
    if len(shape) == 0:
        assert param.size == 1, "Unexpected situation happend!"
        shape = [1]
    assert str(param.dtype) in dtype_map, "Unknown dtype of params."

    fp = open(os.path.join(dir, param_name), 'wb')
    fp.write(struct.pack('i', 0))
    fp.write(struct.pack('L', 0))
    fp.write(struct.pack('i', 0))
    tensor_desc = framework_pb2.VarType.TensorDesc()
    tensor_desc.data_type = dtype_map[str(param.dtype)][0]
    tensor_desc.dims.extend(shape)
    desc_size = tensor_desc.ByteSize()
    fp.write(struct.pack('i', desc_size))
    fp.write(tensor_desc.SerializeToString())
    param.tofile(fp)
    fp.close()
