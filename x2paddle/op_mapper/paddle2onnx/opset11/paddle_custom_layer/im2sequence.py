import onnx
import numpy as np
from onnx import onnx_pb, helper
from x2paddle.op_mapper.paddle2onnx.opset10.paddle_custom_layer.im2sequence import im2sequence as im2sequence10


def im2sequence(op, block):
    return im2sequence10(op, block)
