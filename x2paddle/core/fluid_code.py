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


class Layer(object):
    def __init__(self):
        self.op = None
        self.param_attr = dict()
        self.input = None
        self.output = None
        self.str_code = None

    def get_code(self):
        if self.str_code is not None:
            return self.str_code


class FluidCode(object):
    def __init__(self):
        self.codes = list()

    def add_layer(self, op, input, output, param_attr=None):
        
