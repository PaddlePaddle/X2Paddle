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
from .utils import *


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


def spectral_norm(module,
                  name='weight',
                  n_power_iterations=1,
                  eps=1e-12,
                  dim=None):
    input = getattr(module, name)
    if dim is None:
        if isinstance(module,
                      (paddle.nn.Conv1DTranspose, pdddle.nn.Conv2DTranspose,
                       paddle.nn.Conv3DTranspose)):
            dim = 1
        else:
            dim = 0
    t = str(input.dtype).lower().strip().split(".")[-1]
    t = TYPE_MAPPER[t]
    spectral_norm = paddle.nn.SpectralNorm(
        input.shape, dim=dim, power_iters=n_power_iterations, eps=eps, dtype=t)
    out = spectral_norm(input)
    return out
