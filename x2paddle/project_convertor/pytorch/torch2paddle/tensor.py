# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import paddle
import copy
from functools import reduce
from functools import partial


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


@add_tensor_function
def fill_(self, value):
    paddle.assign(
        paddle.full_like(
            self, value, dtype="float32").cast(self.dtype),
        output=self)


pd_sum = partial(paddle.Tensor.sum)


@add_tensor_function
def sum(self, dim, keepdim=False, dtype=None):
    return pd_sum(self, axis=dim, dtype=dtype, keepdim=keepdim)


pd_sort = partial(paddle.Tensor.sort)


@add_tensor_function
def sort(self, dim=-1, descending=False, out=None):
    return pd_sort(
        self, axis=dim, descending=descending), paddle.argsort(
            self, axis=dim, descending=descending)


pd_reshape = partial(paddle.Tensor.reshape)


@add_tensor_function
def reshape(self, *shape):
    return pd_reshape(self, shape)


pd_transpose = partial(paddle.Tensor.transpose)


@add_tensor_function
def transpose(self, dim0, dim1=None):
    if dim1 is None:
        return pd_transpose(self, dim0)
    else:
        shape = self.shape
        perm = list(range(len(shape)))
        dim0 = (dim0 + len(shape)) if dim0 < 0 else dim0
        dim1 = (dim1 + len(shape)) if dim1 < 0 else dim1
        perm[dim0] = dim1
        perm[dim1] = dim0
        return pd_transpose(self, perm)


pd_max = partial(paddle.Tensor.max)


@add_tensor_function
def max(self, dim, keepdim=None):
    return pd_max(self, dim, keepdim), paddle.argmax(self, dim, keepdim)


pd_min = partial(paddle.Tensor.min)


@add_tensor_function
def min(self, dim, keepdim=None):
    return pd_min(self, dim, keepdim), paddle.argmin(self, dim, keepdim)


pd_expand = partial(paddle.Tensor.expand)


@add_tensor_function
def expand(self, *sizes):
    return pd_expand(self, sizes)


@add_tensor_function
def div(self, value):
    return self / value


@add_tensor_function
def eq(self, other):
    return self.equal(other)


@add_tensor_function
def eq_(self, other):
    return self.equal(other)


@add_tensor_function
def mul(self, value):
    return self * value


@add_tensor_function
def mul_(self, value):
    return self * value


pd_cuda = partial(paddle.Tensor.cuda)


@add_tensor_function
def cuda(self, device=None, non_blocking=False, memory_format=None):
    return self


@add_tensor_function
def copy_(self, src, non_blocking=False):
    src = paddle.expand(src, self.shape)
    paddle.assign(src, self)
