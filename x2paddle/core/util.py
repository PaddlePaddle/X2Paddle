# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import numpy
import math
import os


def string(param):
    return "\'{}\'".format(param)

def name_generator(nn_name, nn_name2id):
    if nn_name in nn_name2id:
        nn_name2id[nn_name] += 1
    else:
        nn_name2id[nn_name] = 0
    real_nn_name = nn_name + str(nn_name2id[nn_name])
    return real_nn_name