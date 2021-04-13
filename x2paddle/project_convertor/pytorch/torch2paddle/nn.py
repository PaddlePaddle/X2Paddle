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


class BatchNorm1D(paddle.nn.BatchNorm1D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


class BatchNorm2D(paddle.nn.BatchNorm2D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


class BatchNorm3D(paddle.nn.BatchNorm3D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


class BCEWithLogitsLoss(paddle.nn.BCEWithLogitsLoss):
    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 pos_weight=None):
        super().__init__(weight, reduction=reduction, pos_weight=pos_weight)


class Conv1D(paddle.nn.Conv1D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias_attr=bias if not bias else None)


class Conv2D(paddle.nn.Conv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias_attr=bias if not bias else None)


class Conv3D(paddle.nn.Conv3D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias_attr=bias if not bias else None)


class Conv2DTranspose(paddle.nn.Conv2DTranspose):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros'):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
            bias_attr=bias if not bias else None)
        assert padding_mode == 'zeros', "The padding_mode must be zero in Conv2DTranspose."


class CrossEntropyLoss(paddle.nn.CrossEntropyLoss):
    def __init__(self,
                 weight=None,
                 size_average=None,
                 ignore_index=-100,
                 reduce=None,
                 reduction='mean'):
        super().__init__(weight, reduction=reduction, ignore_index=ignore_index)


class Dropout(paddle.nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(p)


class Embedding(paddle.nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.0,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            sparse=sparse)
        assert max_norm is None, "The max_norm must be None in Embedding!"
        assert not scale_grad_by_freq, "The scale_grad_by_freq must False None in Embedding!"


class InstanceNorm2D(paddle.nn.InstanceNorm2D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=False,
                 track_running_stats=False):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


class Linear(paddle.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            in_features, out_features, bias_attr=bias if not bias else None)


class MaxPool1D(paddle.nn.MaxPool1D):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False):
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            return_mask=return_indices)
        assert dilation == 1, "The dilation must be 1 in MaxPool1D."


class MaxPool2D(paddle.nn.MaxPool2D):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False):
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            return_mask=return_indices)
        assert dilation == 1, "The dilation must be 1 in MaxPool2D."


class MaxPool3D(paddle.nn.MaxPool3D):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False):
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            return_mask=return_indices)
        assert dilation == 1, "The dilation must be 1 in MaxPool3D."


class Softmax(paddle.nn.Softmax):
    def __init__(self, dim=None):
        super().__init__(axis=dim)
