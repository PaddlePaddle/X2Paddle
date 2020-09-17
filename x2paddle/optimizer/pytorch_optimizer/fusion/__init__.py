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
from .constant_fuser import ConstantFuser
from .constant_fuse_pass import ConstantFusePass
from .dropout_fuser import DropoutFuser
from .dropout_fuse_pass import DropoutFusePass
from .fc_fuser import FcFuser
from .fc_fuse_pass import FcFusePass
from .interpolate_bilinear_fuser import InterpolateBilinearFuser
from .interpolate_bilinear_fuse_pass import InterpolateBilinearFusePass
from .reshape_fuser import ReshapeFuser
from .reshape_fuse_pass import ReshapeFusePass
