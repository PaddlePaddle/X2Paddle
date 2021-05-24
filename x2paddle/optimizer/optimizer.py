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

from x2paddle.optimizer.pass_manager import PassManager
from x2paddle.optimizer.fusion import *
from x2paddle.optimizer.elimination import *


class GraphOptimizer(object):
    def __init__(self, source_frame, jit_type="trace"):
        if source_frame == "pytorch":
            if jit_type == "trace":
                self.passes = ["trace_fc_fuse_pass"]
            else:
                self.passes = [
                    "constant_fuse_pass", "batchnorm2d_fuse_pass",
                    "interpolate_bilinear_fuse_pass", "fc_fuse_pass",
                    "adaptive_pool2d_fuse_pass", "reshape_fuse_pass",
                    "dropout_fuse_pass", "if_fuse_pass"
                ]
        elif source_frame == "caffe":
            self.passes = ["bn_scale_fuse_pass"]
        elif source_frame == "tf":
            self.passes = [
                "conv2d_add_fuse_pass", "tf_batchnorm_fuse_pass",
                "prelu_fuse_pass", "transpose_eliminate_pass"
            ]
        else:
            self.passes = []

    def optimize(self, graph):
        for pass_name in self.passes:
            pass_ = PassManager.lookup(pass_name)()
            if pass_name.endswith("_eliminate_pass") or pass_name.endswith(
                    "conv2d_add_fuse_pass"):
                pass_.apply(graph)
            else:
                while True:
                    before_len = len(graph.layers)
                    pass_.apply(graph)
                    after_len = len(graph.layers)
                    if before_len == after_len:
                        break
            print("{} done!".format(pass_name))
        return graph
