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

import math
import sys
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
import warnings
from onnx import helper, onnx_pb
from x2paddle.op_mapper.paddle2onnx.opset10.paddle_custom_layer.multiclass_nms import multiclass_nms as multiclass_nms10


def multiclass_nms(op, block):
    """
    Convert the paddle multiclass_nms to onnx op.
    This op is get the select boxes from origin boxes.
    """
    return multiclass_nms10(op, block)
