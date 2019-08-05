#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import numpy
import math
import os


def string(param):
    return "\'{}\'".format(param)


# This func will copy to generate code file
def run_net(param_dir="./"):
    import os
    inputs, outputs = x2paddle_net()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        b = os.path.exists(os.path.join(param_dir, var.name))
        return b

    fluid.io.load_vars(exe,
                       param_dir,
                       fluid.default_main_program(),
                       predicate=if_exist)
