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
import paddle.fluid as fluid
import numpy
import math
import os


def string(param):
    return "\'{}\'".format(param)


def color_log(log_str):
    try:
        from colorama import init, Fore
        init(autoreset=True)
        print(Fore.RED + log_str)
    except:
        print(log_str)


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
    if not os.path.exists(dir):
        os.makedirs(dir)

    fp = open(os.path.join(dir, param_name), 'wb')
    numpy.array([0], dtype='int32').tofile(fp)
    numpy.array([0], dtype='int64').tofile(fp)
    numpy.array([0], dtype='int32').tofile(fp)
    tensor_desc = framework_pb2.VarType.TensorDesc()
    tensor_desc.data_type = dtype_map[str(param.dtype)][0]
    tensor_desc.dims.extend(shape)
    desc_size = tensor_desc.ByteSize()
    numpy.array([desc_size], dtype='int32').tofile(fp)
    fp.write(tensor_desc.SerializeToString())
    param.tofile(fp)
    fp.close()


def init_net(param_dir="./"):
    import os
    exe = fluid.Executor(fluid.CUDAPlace(0))
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        b = os.path.exists(os.path.join(param_dir, var.name))
        return b

    fluid.io.load_vars(exe,
                       param_dir,
                       fluid.default_main_program(),
                       predicate=if_exist)
