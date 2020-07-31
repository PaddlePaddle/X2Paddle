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
import x2paddle
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from x2paddle.op_mapper.paddle2onnx.opset9.opset import OpSet9


class OpSet10(OpSet9):
    def __init__(self):
        super(OpSet10, self).__init__()

    def slice(self, op, block):
        axes = op.attr('axes')
        starts = op.attr('starts')
        ends = op.attr('ends')
        axes_name = self.get_name(op.type, 'axes')
        starts_name = self.get_name(op.type, 'starts')
        ends_name = self.get_name(op.type, 'ends')

        axes_node = self.make_constant_node(axes_name,
                                            onnx_pb.TensorProto.INT64, axes)
        starts_node = self.make_constant_node(starts_name,
                                              onnx_pb.TensorProto.INT64, starts)
        ends_node = self.make_constant_node(ends_name,
                                            onnx_pb.TensorProto.INT64, ends)
        node = helper.make_node(
            "Slice",
            inputs=[op.input('Input')[0], starts_name, ends_name, axes_name],
            outputs=op.output('Out'), )
        return [starts_node, ends_node, axes_node, node]

    def im2sequence(self, op, block):
        from .paddle_custom_layer.im2sequence import im2sequence
        return im2sequence(op, block)

    def yolo_box(self, op, block):
        from .paddle_custom_layer.yolo_box import yolo_box
        return yolo_box(op, block)

    def multiclass_nms(self, op, block):
        from .paddle_custom_layer.multiclass_nms import multiclass_nms
        return multiclass_nms(op, block)

    def box_coder(self, op, block):
        from .paddle_custom_layer.box_coder import box_coder
        return box_coder(op, block)

    def prior_box(self, op, block):
        from .paddle_custom_layer.prior_box import prior_box
        return prior_box(op, block)
