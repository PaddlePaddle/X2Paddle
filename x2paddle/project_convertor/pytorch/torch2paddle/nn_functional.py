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