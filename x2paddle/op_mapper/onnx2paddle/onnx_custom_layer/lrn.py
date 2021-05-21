# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


class LocalResponseNorm(object):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1.):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def __call__(self, x):
        sizes = x.shape
        dim = len(sizes)
        if dim < 3:
            raise ValueError(
                'Expected 3D or higher dimensionality input, but got {} dimensions'.
                format(dim))
        div = paddle.unsqueeze(paddle.multiply(x, x), axis=1)
        pad4d_shape = [self.size // 2, (self.size - 1) // 2, 0, 0]
        pool2d_shape = (1, self.size)
        pad5d_shape = [self.size // 2, (self.size - 1) // 2, 0, 0, 0, 0]
        pool3d_shape = (1, 1, self.size)

        if dim == 3:
            div = paddle.nn.functional.pad(div, pad=pad4d_shape)
            div = paddle.nn.functional.avg_pool2d(
                div, kernel_size=pool2d_shape, stride=1)
            div = paddle.squeeze(div, axis=1)
        else:
            tmp = paddle.unsqueeze(x, axis=1)
            reshape_shape = paddle.shape(tmp)
            new_reshape_shape = paddle.cast(reshape_shape, "float32")
            index = paddle.full(shape=[1], fill_value=-2, dtype="int32")
            value = paddle.full(shape=[1], fill_value=-1, dtype="float32")
            new_reshape_shape = paddle.scatter(new_reshape_shape, index, value)
            new_reshape_shape = paddle.cast(new_reshape_shape, "int32")
            div = paddle.reshape(div, shape=reshape_shape)
            div = paddle.nn.functional.pad(div,
                                           pad=pad5d_shape,
                                           data_format='NCDHW')
            div = paddle.nn.functional.avg_pool3d(
                div, kernel_size=pool3d_shape, stride=1)
            div = paddle.reshape(paddle.squeeze(div, axis=1), sizes)

        div = paddle.scale(div, scale=self.alpha, bias=self.k)
        div = paddle.pow(div, self.beta)
        res = paddle.divide(x, div)
        return res
