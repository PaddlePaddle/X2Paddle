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

from x2paddle.optimizer.pytorch_optimizer.fusion import *
from x2paddle.optimizer.pytorch_optimizer.pass_manager import PassManager


class GraphOptimizer(object):
    def __init__(self):
        self.passes = [
            "constant_fuse_pass", "batchnorm2d_fuse_pass",
            "interpolate_bilinear_fuse_pass", "fc_fuse_pass",
            "adaptive_pool2d_fuse_pass", "reshape_fuse_pass",
            "dropout_fuse_pass"
        ]

    def optimize(self, graph):
        for pass_name in self.passes:
            pass_ = PassManager.lookup(pass_name)()
            pass_.apply(graph)
            print("{} done!".format(pass_name))
        return graph
