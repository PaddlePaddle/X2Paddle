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

from .adaptive_pool2d_fuser import AdaptivePool2dFuser
from .adaptive_pool2d_fuse_pass import AdaptivePool2dFusePass
from .batchnorm2d_fuser import BatchNorm2dFuser
from .batchnorm2d_fuse_pass import BatchNorm2dFusePass
from .bn_scale_fuser import BNScaleFuser
from .bn_scale_fuse_pass import BNScaleFusePass
from .constant_fuser import ConstantFuser
from .constant_fuse_pass import ConstantFusePass
from .conv2d_add_fuser import Conv2DAddFuser
from .conv2d_add_fuse_pass import Conv2DAddFusePass
from .dropout_fuser import DropoutFuser
from .dropout_fuse_pass import DropoutFusePass
from .fc_fuser import FcFuser
from .fc_fuse_pass import FcFusePass
from .if_fuser import IfFuser
from .if_fuse_pass import IfFusePass
from .interpolate_bilinear_fuser import InterpolateBilinearFuser
from .interpolate_bilinear_fuse_pass import InterpolateBilinearFusePass
from .prelu_fuser import PReLUFuser
from .prelu_fuse_pass import PReLUFusePass
from .reshape_fuser import ReshapeFuser
from .reshape_fuse_pass import ReshapeFusePass
from .tf_batchnorm_fuser import TFBatchNormFuser
from .tf_batchnorm_fuse_pass import TFBatchNormFusePass
from .trace_fc_fuser import TraceFcFuser
from .trace_fc_fuse_pass import TraceFcFusePass
