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


class Emitter(object):
    def __init__(self):
        self.paddle_codes = ""
        self.tab = "    "

    def add_codes(self, codes, indent=0):
        if isinstance(codes, list):
            for code in codes:
                self.paddle_codes += (self.tab * indent + code + '\n')
        elif isinstance(codes, str):
            self.paddle_codes += (self.tab * indent + codes + '\n')
        else:
            raise Exception("Unknown type of codes")

    def add_heads(self):
        self.add_codes("import paddle.fluid as fluid")
        self.add_codes("")

    def save_inference_model(self):
        print("Not Implement")

    def save_python_code(self):
        print("Not Implement")
