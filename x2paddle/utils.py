# -*- coding:UTF-8 -*-
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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


def string(param):
    """ 生成字符串。
    """
    return "\'{}\'".format(param)


import paddle
version = paddle.__version__
v0, v1, v2 = version.split('.')
is_larger_21 = True
if not ((v0 == '0' and v1 == '0' and v2 == '0') or
        (int(v0) >= 2 and int(v1) >= 1)):
    is_larger_21 = False

if is_larger_21:
    pd_float16 = paddle.float16
    pd_float32 = paddle.float32
    pd_float64 = paddle.float64
    pd_uint8 = paddle.uint8
    pd_int8 = paddle.int8
    pd_int16 = paddle.int16
    pd_int32 = paddle.int32
    pd_int64 = paddle.int64
    pd_bool = paddle.bool
else:
    from paddle.fluid.core import VarDesc
    pd_float16 = VarDesc.VarType.FP16
    pd_float32 = VarDesc.VarType.FP32
    pd_float64 = VarDesc.VarType.FP64
    pd_uint8 = VarDesc.VarType.UINT8
    pd_int8 = VarDesc.VarType.INT8
    pd_int16 = VarDesc.VarType.INT16
    pd_int32 = VarDesc.VarType.INT32
    pd_int64 = VarDesc.VarType.INT64
    pd_bool = VarDesc.VarType.BOOL
