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

from x2paddle.core.util import *
import os


class Emitter(object):
    def __init__(self):
        self.paddle_codes = ""
        self.tab = "    "
        self.net_code = list()
        self.weights = dict()

    def add_codes(self, codes, indent=0):
        if isinstance(codes, list):
            for code in codes:
                self.paddle_codes += (self.tab * indent + code + '\n')
        elif isinstance(codes, str):
            self.paddle_codes += (self.tab * indent + codes + '\n')
        else:
            raise Exception("Unknown type of codes")

    def add_heads(self):
        self.add_codes("from paddle.fluid.initializer import Constant")
        self.add_codes("from paddle.fluid.param_attr import ParamAttr")
        self.add_codes("import paddle.fluid as fluid")
        self.add_codes("")

    def save_inference_model(self):
        print("Not Implement")

    def save_python_model(self, save_dir):
        for name, param in self.weights.items():
            export_paddle_param(param, name, save_dir)
        self.add_heads()
        self.add_codes(self.net_code)
        fp = open(os.path.join(save_dir, "model.py"), 'w')
        fp.write(self.paddle_codes)
        fp.close()
