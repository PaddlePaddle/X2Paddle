import paddle
import copy

TYPE_ORDER = ["bool", "int32", "int64", "float32", "float64"]
TYPE_MAPPER = {"fp32": "float32",
               "fp64": "float64"}

def binary_cross_entropy_with_logits(input, 
                                     target, 
                                     weight=None, 
                                     size_average=None, 
                                     reduce=None, 
                                     reduction='mean', 
                                     pos_weight=None):
    if not reduce or not size_average:
        reduction = "sum"
    input_t = str(input.dtype).lower().strip().split(".")[-1]
    if input_t in TYPE_MAPPER:
        input_t = TYPE_MAPPER[input_t]
    input_index = TYPE_ORDER.index(input_t)
    target_t = str(target.dtype).lower().strip().split(".")[-1]
    if target_t in TYPE_MAPPER:
        target_t = TYPE_MAPPER[target_t]
    target_index = TYPE_ORDER.index(target_t)
    if input_index < target_index:
        real_type = TYPE_ORDER[target_index]
        input = input.cast(real_type)
    else:
        real_type = TYPE_ORDER[input_index]
        target = target.cast(real_type)
    return paddle.nn.functional.binary_cross_entropy_with_logits(
        input, target, weight, reduction, pos_weight)


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return paddle.nn.functional.avg_pool1d(input, kernel_size, stride=stride, 
                                           padding=padding, ceil_mode=ceil_mode, 
                                           exclusive=not count_include_pad)
    
def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, 
               count_include_pad=True, divisor_override=None):
    return paddle.nn.functional.avg_pool2d(input, kernel_size, stride=stride, 
                                           padding=padding, ceil_mode=ceil_mode, 
                                           exclusive=not count_include_pad, 
                                           divisor_override=divisor_override)

def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, 
               count_include_pad=True, divisor_override=None):
    return paddle.nn.functional.avg_pool3d(input, kernel_size, stride=stride, 
                                           padding=padding, ceil_mode=ceil_mode, 
                                           exclusive=not count_include_pad, 
                                           divisor_override=divisor_override)

def dropout(input, p=0.5, training=True, inplace=False):
    return paddle.nn.functional.dropout(input, p=p, training=training)

def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    return paddle.nn.functional.log_softmax(input, axis=dim, dtype=None)

def relu(input, inplace=False):
    return paddle.nn.functional.relu(input)

def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0):
    paddle.nn.functional.smooth_l1_loss(input, target, reduction=reduction, delta=beta)

def softmax(input, dim=None, _stacklevel=3, dtype=None):
    return paddle.nn.functional.softmax(input, axis=dim, dtype=dtype)
