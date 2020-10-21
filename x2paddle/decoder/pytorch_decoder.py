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

import os
import sys
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
    def __init__(self, script_path=None):
        self.script = torch.jit.load(script_path)
        self.graph = self._optimize_graph(self.script.inlined_graph)
            
class TraceDecoder(Decoder):
    """ PyTorchModule后使用trace方式转换为ScriptModule。
        
        Args:
            model_path (str): PyTorchModule保存路径。
            input_files (list): 输入网络的numpy，每个numpy保存成.npy文件, 
                                文件路径存储在input_files中。
    """
    def __init__(self, model_path, input_files=list()):
        # TODO(syf): 传入pytorch的Module(即import)，否则出错
        model = torch.load(model_path)
        model.eval()
        input_list = list()
        for npy_file in input_files:
            input_list.append(torch.tensor(np.load(npy_file)))
        self.script = torch.jit.trace(model, input_list, strict=False)
        self.graph = self._optimize_graph(self.script.inlined_graph)
#         print(self.graph)
#         print(getattr(getattr(self.script.decoder.block, "5").layer, "2"))
