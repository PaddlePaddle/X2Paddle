import onnx
import numpy as np
from onnx import onnx_pb, helper
from x2paddle.op_mapper.paddle2onnx.opset9.paddle_custom_layer.im2sequence import im2sequence as im2sequence9


def im2sequence(op, block):
    return im2sequence9(op, block)
