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

import torch
import numpy as np


class Decoder(object):
     def _optimize_graph(self, graph):
        torch._C._jit_pass_constant_propagation(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_peephole(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_constant_propagation(graph)
        return graph       


class ScriptDecoder(Decoder):
    """ 当script_path非None，直接load ScriptModule;
        当model_path非None，load PyTorchModule后使用script方式转换为ScriptModule。
        
        Args:
            script_path (str): ScriptModule保存路径。
            model_path (str): PyTorchModule保存路径。
    """
    def __init__(self, script_path=None, model_path=None):
        if script_path is not None:
            self.script = torch.jit.load(script_path)
        else:
            if model_path is not None:
                model = torch.load(model_path)
                self.script = torch.jit.script(model)
            else:
                raise Exception("The script_path or model_path must be defined!")
        self.graph = self._optimize_graph(self.script.inlined_graph)
            
class TraceDecoder(Decoder):
    """ PyTorchModule后使用trace方式转换为ScriptModule。
        
        Args:
            model_path (str): PyTorchModule保存路径。
            input_file_list (list): 输入网络的numpy，每个numpy保存成.npy文件, 
                                    文件路径存储在input_file_list中。
    """
    def __init__(self, model_path, input_file_list=list()):
        model = torch.load(model_path)
        input_list = list()
        for npy_file in input_file_list:
            input_list.append(torch.Tensor(np.load(npy_file)))
        self.script = torch.jit.trace(model, input_list)
        self.graph = self._optimize_graph(self.script.inlined_graph)
            
