# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from x2paddle.op_mapper.onnx.opset9 import ONNXOpMapperOpSet9


class ONNXOpMapperFactory:
    def __init__(self):
        self.support_op_sets = [9, ]
        self.default_op_set = 9

    def create_onnx_op_mapper(self, decoder):
        run_op_set = self.default_op_set
        OpMapper = ''
        if decoder.op_set in self.support_op_sets:
            OpMapper = 'ONNXOpMapperOpSet' + str(decoder.op_set)
        elif decoder.op_set < self.default_op_set:
            OpMapper = 'ONNXOpMapperOpSet' + str(self.default_op_set)
        else:
            for op_set in self.support_op_sets:
                if decoder.op_set > op_set:
                    run_op_set = op_set
                else:
                    break
            OpMapper = 'ONNXOpMapperOpSet' + str(run_op_set)
        print(
            'Now, onnx2paddle support convert onnx model opset_verison {},'
            'opset_verison of your onnx model is {}, automatically treated as op_set: {}.'
            .format(self.support_op_sets, decoder.op_set, run_op_set))
        return eval(OpMapper)(decoder)
