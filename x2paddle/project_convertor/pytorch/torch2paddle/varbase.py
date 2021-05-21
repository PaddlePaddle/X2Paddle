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
from paddle.fluid.core import VarBase
from x2paddle.utils import paddle_dtypes


def is_condition_one(idx):
    """
    a = paddle.to_tensor(np.array([[1,2,3], [4,5,6]]).astype("float32"))
    mask = paddle.to_tensor(np.array([True, False]).astype("bool"))
    a[mask, :]
    a[mask, ...]
    """
    if not (isinstance(idx[0], paddle.Tensor) and \
            idx[0].dtype == paddle_dtypes.t_bool):
        return False
    if len(idx) == 1:
        return False
    if len(idx) > 1:
        if idx[1] is Ellipsis:
            return True
        for ele in idx[1:]:
            if isinstance(
                    ele, slice
            ) and ele.start is None and ele.start is None and ele.step is None:
                continue
            else:
                return False
    return True


def is_condition_two(idx):
    """
    a = paddle.to_tensor(np.random.rand(1, 2, 3).astype("float32"))
    a[..., :2]
    """
    if idx[0] is Ellipsis and (isinstance(idx[1], slice) or isinstance(idx[1],
                                                                       int)):
        return True
    return False


VarBase.tmp = VarBase.__getitem__


def __getitem__(self, idx):
    is_bool = False
    if self.dtype == paddle_dtypes.t_bool:
        self = self.cast("int32")
        is_bool = True
    if isinstance(idx, paddle.Tensor) and len(idx.shape) == 1:
        out = paddle.gather(self, idx)
        return out.cast("bool") if is_bool else out
    elif isinstance(idx, paddle.Tensor) and idx.dtype == paddle_dtypes.t_bool:
        idx = paddle.cast(idx, "int32")
        idx = paddle.nonzero(idx)
        out = paddle.gather_nd(self, idx)
        return out.cast("bool") if is_bool else out
    elif isinstance(idx, tuple):
        if is_condition_one(idx):
            first_idx = idx[0]
            first_idx = paddle.cast(first_idx, "int32")
            first_idx = paddle.nonzero(first_idx)
            out = paddle.gather_nd(self, first_idx)
            return out.cast("bool") if is_bool else out
        elif is_condition_two(idx):
            new_idx = list()
            for i in range(len(self.shape) - 1):
                new_idx.append(slice(None, None, None))
            new_idx.append(list(idx)[-1])
            out = self.tmp(tuple(new_idx))
            return out.cast("bool") if is_bool else out
        else:
            out = self.tmp(idx)
            return out.cast("bool") if is_bool else out
        # TODO(syf): 出来为(slice(None, None, None), slice(None, None, None), 0)
    else:
        out = self.tmp(idx)
        if out.shape == [1]:
            return out.numpy()[0]
        else:
            return out


VarBase.__getitem__ = __getitem__

VarBase.setitem_tmp = VarBase.__setitem__


def __setitem__(self, idx, value):
    if isinstance(idx, paddle.Tensor) and idx.dtype == paddle_dtypes.t_bool:
        """
        a = paddle.to_tensor(np.array([1,2,3]).astype("float32"))
        mask = paddle.to_tensor(np.array([True, False, True]).astype("bool"))
        a[mask] = 1
        """
        value_tensor = paddle.full(self.shape, value, self.dtype)
        paddle.assign(paddle.where(idx, value_tensor, self), self)
    else:
        return self.setitem_tmp(idx, value)


VarBase.__setitem__ = __setitem__
