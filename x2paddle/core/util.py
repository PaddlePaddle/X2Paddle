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


def string(param):
    return "\'{}\'".format(param)


def get_same_padding(input_size, kernel_size, stride):
    new_size = int(math.ceil(input_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + filter_size - in_size
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0
    return [pad0, pad1]
