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


class AvgPool1D(paddle.nn.AvgPool1D):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 count_include_pad=True,
                 divisor_override=None):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=padding,
            exclusive=count_include_pad,
            divisor_override=divisor_override)


class AvgPool2D(paddle.nn.AvgPool2D):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 count_include_pad=True,
                 divisor_override=None):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=padding,
            exclusive=count_include_pad,
            divisor_override=divisor_override)


class AvgPool3D(paddle.nn.AvgPool3D):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 count_include_pad=True,
                 divisor_override=None):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=padding,
            exclusive=count_include_pad,
            divisor_override=divisor_override)


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


@property
def in_channels(self):
    return self._in_channels


setattr(paddle.nn.layer.conv._ConvNd, "in_channels", in_channels)


@property
def out_channels(self):
    return self._out_channels


setattr(paddle.nn.layer.conv._ConvNd, "out_channels", out_channels)


@property
def kernel_size(self):
    return self._kernel_size


setattr(paddle.nn.layer.conv._ConvNd, "kernel_size", kernel_size)


@property
def stride(self):
    return self._stride


setattr(paddle.nn.layer.conv._ConvNd, "stride", stride)


@property
def padding(self):
    return self._padding


setattr(paddle.nn.layer.conv._ConvNd, "padding", padding)


@property
def dilation(self):
    return self._dilation


setattr(paddle.nn.layer.conv._ConvNd, "dilation", dilation)


@property
def groups(self):
    return self._groups


setattr(paddle.nn.layer.conv._ConvNd, "groups", groups)



class ConstantPad2D(paddle.nn.Pad2D):
    def __init__(self, padding, value):
        super().__init__(padding,
                         value=value)


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
        
        
class KLDivLoss(paddle.nn.Layer):
    def __init__(self,
                 size_average=None, 
                 reduce=None, 
                 reduction='mean', 
                 log_target=False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
    
    def forward(self, input, target):
        if self.log_target:
            out = paddle.exp(target) * (target - input)
        else:
            out_pos = target * (paddle.log(target) - input)
            zeros = paddle.zeros_like(output_pos)
            out = paddle.where(target > 0, out_pos, zeros)
        out_sum = paddle.sum(out)
        if self.reduction == "sum":
            return out_sum
        elif self.reduction == "batchmean":
            n = input.shape[0]
            return out_sum / n
        elif self.reduction == "mean":
            return paddle.mean(out)
        else:
            return out


class Linear(paddle.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            in_features, out_features, bias_attr=bias if not bias else None)
        
        
class L1Loss(paddle.nn.L1Loss):
    def __init__(self,
                 size_average=None, reduce=None, reduction='mean'):
        super().__init__(reduction=reduction)



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
        
        
class MaxUnpool2D(paddle.nn.Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(stride, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            if isinstance(stride, int):
                self.stride = (stride, stride)
            else:
                self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
    def forward(self, input, indices, output_size=None):
        if output_size is None:
            n, c, h, w = x.shape
            out_h = (h - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            out_w = (w - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
            output_size = (n, c, out_h, out_w)
        t = str(x.dtype).lower().strip().split(".")[-1]
        t = TYPE_MAPPER[t]
        out = paddle.zeros(output_size, dtype=t)
        flatten_out = paddle.flatten(out)
        flatten_indices = paddle.flatten(indices)
        flatten_input = paddle.flatten(input)
        for i in range(flatten_indices.shape[0]):
            flatten_out[flatten_indices[i]] = flatten_input[flatten_indices[i]]
        out = paddle.reshape(flatten_out, out.shape)
        return out
    
    
class ReflectionPad2D(paddle.nn.Pad2D):
    def __init__(self, padding):
        super().__init__(padding,
                         mode="reflect")
        
        
class ReplicationPad2D(paddle.nn.Pad2D):
    def __init__(self, padding):
        super().__init__(padding,
                         mode="replicate")


class Softmax(paddle.nn.Softmax):
    def __init__(self, dim=None):
        super().__init__(axis=dim)
        
        
class SyncBatchNorm(paddle.nn.SyncBatchNorm):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 process_group=None):
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
        
        
class ZeroPad2D(paddle.nn.Pad2D):
    def __init__(self, padding):
        super().__init__(padding)
