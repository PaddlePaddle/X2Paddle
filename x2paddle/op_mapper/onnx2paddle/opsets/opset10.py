# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from x2paddle.core.graph import GraphNode
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.fluid_code import Layer
from x2paddle.core.fluid_code import FluidCode
from x2paddle.decoder.onnx_decoder import ONNXGraph, ONNXGraphNode, ONNXGraphDataNode
from x2paddle.op_mapper.onnx.custom_layer import *
from x2paddle.op_mapper.onnx.opset9 import ONNXOpMapperOpSet9
from x2paddle.core.util import string
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import logging as _logging
from collections import OrderedDict
import math
import os
import shutil
from functools import reduce


class ONNXOpMapperOpSet10(ONNXOpMapperOpSet9):
    def __init__(self, decoder):
        super(ONNXOpMapperOpSet10, self).__init__(decoder)
