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

from .adaptive_pool2d_fuser import Dygraph_AdaptivePool2dFuser
from .adaptive_pool2d_fuse_pass import Dygraph_AdaptivePool2dFusePass
from .batchnorm2d_fuser import Dygraph_BatchNorm2dFuser
from .batchnorm2d_fuse_pass import Dygraph_BatchNorm2dFusePass
from .bn_scale_fuser import Dygraph_BNScaleFuser
from .bn_scale_fuse_pass import Dygraph_BNScaleFusePass
from .constant_fuser import Dygraph_ConstantFuser
from .constant_fuse_pass import Dygraph_ConstantFusePass
from .dropout_fuser import Dygraph_DropoutFuser
from .dropout_fuse_pass import Dygraph_DropoutFusePass
from .fc_fuser import Dygraph_FcFuser
from .fc_fuse_pass import Dygraph_FcFusePass
from .interpolate_bilinear_fuser import Dygraph_InterpolateBilinearFuser
from .interpolate_bilinear_fuse_pass import Dygraph_InterpolateBilinearFusePass
from .reshape_fuser import Dygraph_ReshapeFuser
from .reshape_fuse_pass import Dygraph_ReshapeFusePass
