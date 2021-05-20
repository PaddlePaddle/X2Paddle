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

import math
from functools import reduce
import paddle
from paddle.fluid import framework
from paddle.fluid.core import VarDesc
from paddle.fluid.initializer import XavierInitializer, MSRAInitializer
from paddle.fluid.data_feeder import check_variable_and_dtype
from x2paddle.utils import paddle_dtypes


def _calculate_fan_in_and_fan_out(var):
    dimensions = var.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for var with fewer than 2 dimensions"
        )
    num_input_fmaps = var.shape[0]
    num_output_fmaps = var.shape[1]
    receptive_field_size = 1
    if var.dim() > 2:
        receptive_field_size = reduce(lambda x, y: x * y, var.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(var, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(
            mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(var)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_gain(nonlinearity, param=None):
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d'
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(
                param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(
                param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


class KaimingNormal(MSRAInitializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingNormal, self).__init__(uniform=False, fan_in=None, seed=0)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def __call__(self, var, block=None):
        """Initialize the input tensor with MSRA initialization.
        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.
        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        f_in, f_out = self._compute_fans(var)

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == paddle_dtypes.t_float16:
            out_dtype = paddle_dtypes.t_float32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['masra_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        fan = _calculate_correct_fan(var, self.mode)
        gain = _calculate_gain(self.nonlinearity, self.a)
        std = gain / math.sqrt(fan)
        op = block._prepend_op(
            type="gaussian_random",
            outputs={"Out": out_var},
            attrs={
                "shape": out_var.shape,
                "dtype": int(out_dtype),
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


def kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=KaimingNormal(
            a=a, mode=mode, nonlinearity=nonlinearity))
    paddle.assign(param, replaced_param)


class XavierNormal(XavierInitializer):
    def __init__(self, gain=1.0):
        super(XavierNormal, self).__init__(
            uniform=True, fan_in=None, fan_out=None, seed=0)
        self._gain = gain

    def __call__(self, var, block=None):
        block = self._check_block(block)
        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out", ["float16", "float32", "float64"],
                                 "xavier_init")

        fan_in, fan_out = _calculate_fan_in_and_fan_out(var)

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == paddle_dtypes.t_float16:
            out_dtype = paddle_dtypes.t_float32
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

        std = self._gain * math.sqrt(2.0 / float(fan_in + fan_out))
        op = block._prepend_op(
            type="uniform_random",
            inputs={},
            outputs={"Out": out_var},
            attrs={
                "shape": out_var.shape,
                "dtype": out_dtype,
                "min": 0,
                "max": std,
                "seed": self._seed
            },
            stop_gradient=True)
        if var.dtype == paddle_dtypes.t_float16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})
        if not framework.in_dygraph_mode():
            var.op = op
        return op


def xavier_normal_(param, gain=1.0):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=XavierNormal(gain=gain))
    paddle.assign(param, replaced_param)


class XavierUniform(XavierInitializer):
    def __init__(self, gain=1.0):
        super(XavierUniform, self).__init__(
            uniform=True, fan_in=None, fan_out=None, seed=0)
        self._gain = gain

    def __call__(self, var, block=None):
        block = self._check_block(block)
        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out", ["float16", "float32", "float64"],
                                 "xavier_init")

        fan_in, fan_out = _calculate_fan_in_and_fan_out(var)

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == paddle_dtypes.t_float16:
            out_dtype = paddle_dtypes.t_float32
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

        std = self._gain * math.sqrt(2.0 / float(fan_in + fan_out))
        limit = math.sqrt(3.0) * std
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
        if var.dtype == paddle_dtypes.t_float16:
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
    paddle.assign(param, replaced_param)


def constant_init_(param, val):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.full(param.shape, val, param.dtype)))
    paddle.assign(param, replaced_param)


def normal_init_(param, mean=0.0, std=1.0):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.normal(
                mean=mean, std=std, shape=param.shape)))
    paddle.assign(param, replaced_param)


def ones_init_(param):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.ones(param.shape, param.dtype)))
    paddle.assign(param, replaced_param)


def zeros_init_(param):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.zeros(param.shape, param.dtype)))
    paddle.assign(param, replaced_param)
