import onnx
import numpy as np
from onnx import onnx_pb, helper
from x2paddle.op_mapper.paddle2onnx.opset9.paddle_custom_layer.yolo_box import yolo_box as yolo_box9


def yolo_box(op, block):
    return yolo_box9(op, block)
