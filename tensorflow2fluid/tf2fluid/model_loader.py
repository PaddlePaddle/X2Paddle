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
import sys


class ModelLoader(object):
    def __init__(self, model_dir, use_cuda=False):
        sys.path.append(model_dir)
        mymodel = __import__("mymodel")
        self.model = mymodel.Model()
        self.model.build()
        self.inputs = self.model.inputs
        self.outputs = self.model.outputs
        if use_cuda:
            self.exe = fluid.Executor(fluid.CUDAPlace(0))
        else:
            self.exe = fluid.Executor(fluid.CPUPlace())
        self.exe.run(fluid.default_startup_program())

        var_list = list()
        global_block = fluid.default_main_program().global_block()
        with open(model_dir + "/save_var.list") as f:
            for line in f:
                try:
                    var = global_block.var(line.strip())
                    var_list.append(var)
                except:
                    pass
        fluid.io.load_vars(self.exe, model_dir, vars=var_list)
        self.program = fluid.default_main_program()

    def save_inference_model(self, save_dir):
        fluid.io.save_inference_model(save_dir, self.model.inputs,
                                      self.model.outputs, self.exe)

    def inference(self, feed_dict):
        result = self.exe.run(
            self.program, feed=feed_dict, fetch_list=self.model.outputs)
        return result
