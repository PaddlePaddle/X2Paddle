import paddle
import copy
import warnings 
from paddle import DataParallel as Base_DataParallel

TYPE_ORDER = ["bool", "int32", "int64", "float32", "float64"]
TYPE_MAPPER = {"fp32": "float32",
               "fp64": "float64"}

def abs(input, *, out=None):
    return paddle.abs(input)

def arange(start, end, step=1, *, 
           out=None, dtype=None, layout=None, 
           device=None, requires_grad=False):
    return paddle.arange(start, end, step, dtype)

def clip(input, min, max, *, out=None):
    return paddle.clip(input, min, max)
    
def concat(tensors, dim=0):
    x = tensors
    last_index = -1
    for ele in x:
        t = str(ele.dtype).lower().strip().split(".")[-1]
        if t in TYPE_MAPPER:
            t = TYPE_MAPPER[t]
        index = TYPE_ORDER.index(t)
        if last_index < index:
            last_index = index
    real_type = TYPE_ORDER[last_index]
    x = list(x)
    for i in range(len(x)):
        x[i] = x[i].cast(real_type)
    return paddle.concat(x, dim)

def exp(input, *, out=None):
    return paddle.exp(input)

def full(size, fill_value, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if requires_grad:
        return paddle.full(size, fill_value, dtype).requires_grad_(True)
    else:
        return paddle.full(size, fill_value, dtype)

def full_like(input, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if requires_grad:
        return paddle.full_like(input, fill_value, dtype).requires_grad_(True)
    else:
        return paddle.full_like(input, fill_value, dtype)

def load(f, map_location=None, pickle_module=None, **pickle_load_args):
    return paddle.load(f)

def log(input, *, out=None):
    return paddle.log(input)

def logical_and(input, other, *, out=None):
    return paddle.logical_and(input, other, out)

def logical_not(input, *, out=None):
    return paddle.logical_not(input, out)

def logical_or(input, other, *, out=None):
    return paddle.logical_or(input, other, out)

def logical_xor(input, other, *, out=None):
    return paddle.logical_xor(input, other, out)

def matmul(input, other, *, out=None):
    return paddle.matmul(input, other)

def max(input, dim_other=None, keepdim=False, *, out=None):
    if dim_other is None:
        return paddle.max(input)
    elif isinstance(dim_other, paddle.Tensor):
        return paddle.maximum(input, dim_other)
    else:
        return paddle.max(input, axis=dim_other, keepdim=keepdim)

def mean(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        warnings.warn('The output of paddle.mean is not scalar!')
        return paddle.mean(input)
    else:
        return paddle.mean(input, axis=dim, keepdim=keepdim)

def min(input, dim_other=None, keepdim=False, *, out=None):
    if dim_other is None:
        return paddle.min(input)
    elif isinstance(dim_other, paddle.Tensor):
        return paddle.minimum(input, dim_other)
    else:
        return paddle.min(input, axis=dim_other, keepdim=keepdim)

def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        shape = size[0]
    else:
        shape = size
    if requires_grad:
        return paddle.ones(shape, dtype).requires_grad_(True)
    else:
        return paddle.ones(shape, dtype)
    
def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if requires_grad:
        return paddle.ones_like(input, dtype).requires_grad_(True)
    else:
        return paddle.ones_like(input, dtype)
    
def rand(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        shape = size[0]
    else:
        shape = size
    if requires_grad:
        return paddle.rand(shape, dtype).requires_grad_(True)
    else:
        return paddle.rand(shape, dtype)
    
def randperm(n, *, generator=None, out=None, dtype="int64", 
             layout=None, device=None, requires_grad=False, pin_memory=False):
    return paddle.randperm(n, dtype)
    
def save(obj, f, pickle_module=None, pickle_protocol=2):
    return paddle.save(obj, f, pickle_protocol=pickle_protocol)

def sqrt(input, *, out=None):
    return paddle.sqrt(input)

def stack(tensors, dim=0, *, out=None):
    return paddle.stack(tensors, dim)    

def sum(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        warnings.warn('The output of paddle.sum is not scalar!')
        return paddle.sum(input)
    else:
        return paddle.sum(input, axis=dim, keepdim=keepdim)

def unsqueeze(input, dim):
    return paddle.squeeze(input, dim)

def zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        shape = size[0]
    else:
        shape = size
    if requires_grad:
        return paddle.zeros(shape, dtype).requires_grad_(True)
    else:
        return paddle.zeros(shape, dtype)

def zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if requires_grad:
        return paddle.zeros_like(input, dtype).requires_grad_(True)
    else:
        return paddle.zeros_like(input, dtype)
    
    
class DataParallel(Base_DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module)
