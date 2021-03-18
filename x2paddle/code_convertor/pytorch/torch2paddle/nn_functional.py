import paddle
import copy

TYPE_ORDER = ["bool", "int32", "int64", "float32", "float64"]
TYPE_MAPPER = {"fp32": "float32",
               "fp64": "float64"}

def binary_cross_entropy_with_logits(logit, 
                                     label, 
                                     weight=None, 
                                     reduction='mean', 
                                     pos_weight=None):
    logit_t = str(logit.dtype).lower().strip().split(".")[-1]
    if logit_t in TYPE_MAPPER:
        logit_t = TYPE_MAPPER[logit_t]
    logit_index = TYPE_ORDER.index(logit_t)
    label_t = str(label.dtype).lower().strip().split(".")[-1]
    if label_t in TYPE_MAPPER:
        label_t = TYPE_MAPPER[label_t]
    label_index = TYPE_ORDER.index(label_t)
    if logit_index < label_index:
        real_type = TYPE_ORDER[label_index]
        logit = logit.cast(real_type)
    else:
        real_type = TYPE_ORDER[logit_index]
        label = label.cast(real_type)
    return paddle.nn.functional.binary_cross_entropy_with_logits(
        logit, label, weight, reduction, pos_weight)