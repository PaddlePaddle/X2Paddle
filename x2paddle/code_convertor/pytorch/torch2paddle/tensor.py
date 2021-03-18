import paddle
import copy
from functools import reduce
from functools import partial

TYPE_ORDER = ["bool", "int32", "int64", "float32", "float64"]
TYPE_MAPPER = {"fp32": "float32",
               "fp64": "float64"}

def add_tensor_function(func):
    setattr(paddle.Tensor, func.__name__, func)

@property
def data(self):
    return self
setattr(paddle.Tensor, "data", data)

@property
def requires_grad(self):
    return not self.stop_gradient
setattr(paddle.Tensor, "requires_grad", requires_grad)

@add_tensor_function
def requires_grad_(self, requires_grad=True):
    self.stop_gradient = not requires_grad
    return self

@add_tensor_function
def item(self):
    return self.numpy()[0]

@add_tensor_function
def permute(self, *dims):
    return self.transpose(dims)

@add_tensor_function
def clamp_(self, min, max):
    return self.clip(min, max)

@add_tensor_function
def contiguous(self):
    return self

@add_tensor_function
def view(self, *shape):
    return self.reshape(*shape)

@add_tensor_function
def repeat(self, *sizes):
    return self.tile(sizes)

@add_tensor_function
def dim(self):
    return self.ndim

@add_tensor_function
def long(self, memory_format=None):
    return paddle.cast(self, dtype="int64")

@add_tensor_function
def float(self, memory_format=None):
    return paddle.cast(self, dtype="float32")

@add_tensor_function
def cuda(self):
    return self

@add_tensor_function
def size(self, dim=None):
    if dim is not None:
        return self.shape[dim]
    else:
        return self.shape
            

@add_tensor_function
def to(self, *args, **kwargs):
    if len(args) == 1 and "dtype" not in kwargs:
        try:
            return paddle.cast(self, dtype=args[0])
        except Exception:
            return self
    else:
        if len(kwargs) > 0:
            if "dtype" in kwargs:
                return paddle.cast(self, dtype=kwargs["dtype"])
            else:
                return self
        else:
            return self
        
@add_tensor_function
def index_fill_(self, dim, index, val):
    x_shape = self.shape
    index_shape = index.shape
    if dim != 0:
        perm_list = list(range(len(x_shape)))
        while dim < 0:
            dim += len(x_shape)
        perm_list.pop(dim)
        perm_list = [dim] + perm_list
        self = paddle.transpose(self, perm=perm_list)
        s = x_shape.pop(dim)
        x_shape = [s] + x_shape
    updates_shape = index_shape + x_shape[1:]
    updates = paddle.full(updates_shape, fill_value=val, dtype=self.dtype)
    out = paddle.scatter(self, index, updates)
    if dim != 0:
        perm_list = list(range(len(x_shape)))
        perm_list.pop(0)
        perm_list.insert(dim, 0)
        out = paddle.transpose(out, perm=perm_list)
    paddle.assign(out, output=self)


sum_tmp = partial(paddle.Tensor.sum)
@add_tensor_function
def sum(self, dim, keepdim=False, dtype=None):
    return sum_tmp(self, axis=dim, dtype=dtype, keepdim=keepdim)

sort_tmp = partial(paddle.Tensor.sort)
@add_tensor_function
def sort(self, dim=-1, descending=False, out=None):
    return sort_tmp(self, axis=dim, descending=descending), paddle.argsort(self, axis=dim, descending=descending)

reshape_tmp = partial(paddle.Tensor.reshape)
@add_tensor_function
def reshape(self, *shape):
    return reshape_tmp(self, shape)

transpose_tmp = partial(paddle.Tensor.transpose)
@add_tensor_function
def transpose(self, dim0, dim1=None):
    if dim1 is None:
        return transpose_tmp(self, dim0)
    else:
        shape = self.shape
        perm = list(range(len(shape)))
        dim0 = (dim0 + len(shape)) if dim0 < 0 else dim0
        dim1 = (dim1 + len(shape)) if dim1 < 0 else dim1
        perm[dim0] = dim1
        perm[dim1] = dim0
        return transpose_tmp(self, perm)
    
transpose_max = partial(paddle.Tensor.max)
@add_tensor_function
def max(self, dim, keepdim=None):
    return transpose_max(self, dim, keepdim), paddle.argmax(self, dim, keepdim)
    
transpose_min = partial(paddle.Tensor.min)
@add_tensor_function
def min(self, dim, keepdim=None):
    return transpose_min(self, dim, keepdim), paddle.argmin(self, dim, keepdim)
    
def concat(x, axis=0):
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
    return paddle.concat(x, axis)
    