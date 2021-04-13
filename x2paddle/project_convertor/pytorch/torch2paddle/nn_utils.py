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
from paddle.fluid.initializer import XavierInitializer


def clip_grad_value_(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    """
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)


setattr(paddle.nn.utils, "clip_grad_value_", clip_grad_value_)


class XavierUniform(XavierInitializer):
    def __init__(self, gain=1.0, fan_in=None, fan_out=None, name=None):
        super(XavierUniform, self).__init__(
            uniform=True, fan_in=fan_in, fan_out=fan_out, seed=0)
        self._gain = gain

    def __call__(self, var, block=None):
        block = self._check_block(block)
        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out", ["float16", "float32", "float64"],
                                 "xavier_init")

        f_in, f_out = self._compute_fans(var)

        # If fan_in and fan_out are passed, use them
        fan_in = f_in if self._fan_in is None else self._fan_in
        fan_out = f_out if self._fan_out is None else self._fan_out

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['xavier_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if self._uniform:
            limit = self._gain * np.sqrt(6.0 / float(fan_in + fan_out))
            op = block._prepend_op(
                type="uniform_random",
                inputs={},
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                },
                stop_gradient=True)

        else:
            std = np.sqrt(2.0 / float(fan_in + fan_out))
            op = block._prepend_op(
                type="gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                },
                stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


def xavier_uniform_(param, gain=1.0):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=XavierUniform(gain=gain))
    return replaced_param
