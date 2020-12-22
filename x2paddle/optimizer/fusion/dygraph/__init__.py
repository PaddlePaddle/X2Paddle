#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from .adaptive_pool2d_fuser import DygraphAdaptivePool2dFuser
from .adaptive_pool2d_fuse_pass import DygraphAdaptivePool2dFusePass
from .batchnorm2d_fuser import DygraphBatchNorm2dFuser
from .batchnorm2d_fuse_pass import DygraphBatchNorm2dFusePass
from .bn_scale_fuser import DygraphBNScaleFuser
from .bn_scale_fuse_pass import DygraphBNScaleFusePass
from .constant_fuser import DygraphConstantFuser
from .constant_fuse_pass import DygraphConstantFusePass
from .conv2d_add_fuser import DygraphConv2DAddFuser
from .conv2d_add_fuse_pass import DygraphConv2DAddFusePass
from .dropout_fuser import DygraphDropoutFuser
from .dropout_fuse_pass import DygraphDropoutFusePass
from .fc_fuser import DygraphFcFuser
from .fc_fuse_pass import DygraphFcFusePass
from .if_fuser import DygraphIfFuser
from .if_fuse_pass import DygraphIfFusePass
from .interpolate_bilinear_fuser import DygraphInterpolateBilinearFuser
from .interpolate_bilinear_fuse_pass import DygraphInterpolateBilinearFusePass
from .prelu_fuser import DygraphPReLUFuser
from .prelu_fuse_pass import DygraphPReLUFusePass
from .reshape_fuser import DygraphReshapeFuser
from .reshape_fuse_pass import DygraphReshapeFusePass
from .tf_batchnorm_fuser import DygraphTFBatchNormFuser
from .tf_batchnorm_fuse_pass import DygraphTFBatchNormFusePass
from .trace_fc_fuser import TraceFcFuser
from .trace_fc_fuse_pass import TraceFcFusePass
