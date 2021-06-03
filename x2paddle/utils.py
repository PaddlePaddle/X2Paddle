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

import paddle


def string(param):
    """ 生成字符串。
    """
    return "\'{}\'".format(param)


def check_version():
    version = paddle.__version__
    v0, v1, v2 = version.split('.')
    if not ((v0 == '0' and v1 == '0' and v2 == '0') or
            (int(v0) >= 2 and int(v1) >= 1)):
        return False
    else:
        return True


class PaddleDtypes():
    def __init__(self, is_new_version=True):
        if is_new_version:
            self.t_float16 = paddle.float16
            self.t_float32 = paddle.float32
            self.t_float64 = paddle.float64
            self.t_uint8 = paddle.uint8
            self.t_int8 = paddle.int8
            self.t_int16 = paddle.int16
            self.t_int32 = paddle.int32
            self.t_int64 = paddle.int64
            self.t_bool = paddle.bool
        else:
            self.t_float16 = "paddle.fluid.core.VarDesc.VarType.FP16"
            self.t_float32 = "paddle.fluid.core.VarDesc.VarType.FP32"
            self.t_float64 = "paddle.fluid.core.VarDesc.VarType.FP64"
            self.t_uint8 = "paddle.fluid.core.VarDesc.VarType.UINT8"
            self.t_int8 = "paddle.fluid.core.VarDesc.VarType.INT8"
            self.t_int16 = "paddle.fluid.core.VarDesc.VarType.INT16"
            self.t_int32 = "paddle.fluid.core.VarDesc.VarType.INT32"
            self.t_int64 = "paddle.fluid.core.VarDesc.VarType.INT64"
            self.t_bool = "paddle.fluid.core.VarDesc.VarType.BOOL"


is_new_version = check_version()
paddle_dtypes = PaddleDtypes(is_new_version)
