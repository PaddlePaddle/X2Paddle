# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import ast
import astor
import os
import os.path as osp
import shutil
import argparse
from six import text_type as _text_type

from .dependency_analyzer import analyze
from .ast_update import update
from .utils import *


def write_file(path, tree):
    code = astor.to_source(tree)
    code = code.replace("(...)", "...")
    code = add_line_continuation_symbol(code)
    f = open(path, "w")
    f.write(code)
    f.close()


def generate_dependencies(folder_path, file_dependencies):
    for name in os.listdir(folder_path):
        current_path = osp.join(folder_path, name)
        if osp.isfile(current_path) and current_path.endswith(".py"):
            if current_path in file_dependencies:
                continue
            analyze(current_path, file_dependencies)
        elif osp.isdir(current_path):
            generate_dependencies(current_path, file_dependencies)


def convert_code(folder_path, new_folder_path, file_dependencies):
    for name in os.listdir(folder_path):
        current_path = osp.join(folder_path, name)
        new_current_path = osp.join(new_folder_path, name)
        if osp.isfile(current_path) and current_path.endswith(".py"):
            print(current_path)
            root = update(current_path, file_dependencies)
            if root is not None:
                write_file(new_current_path, root)
        elif osp.isdir(current_path):
            if not osp.exists(new_current_path):
                os.makedirs(new_current_path)
            convert_code(current_path, new_current_path, file_dependencies)
        elif osp.isfile(current_path) and osp.splitext(current_path)[
                -1] in [".pth", ".pt", ".ckpt"]:
            continue
        elif osp.isfile(current_path) and current_path.endswith(".pyc"):
            continue
        elif osp.isdir(current_path) and current_path == "__pycache__":
            continue
        elif osp.isdir(current_path) and current_path == ".ipynb_checkpoints":
            continue
        else:
            shutil.copyfile(current_path, new_current_path)


def convert_params(params_path):
    import torch
    import paddle
    params = torch.load(params_path, map_location=torch.device('cpu'))
    new_params = dict()
    bn_w_name_list = list()
    for k, v in params.items():
        if k.endswith(".running_mean"):
            new_params[k.replace(".running_mean", "._mean")] = v.detach().numpy(
            )
        elif k.endswith(".running_var"):
            new_params[k.replace(".running_var", "._variance")] = v.detach(
            ).numpy()
            bn_w_name_list.append(k.replace(".running_var", ".weight"))
        else:
            new_params[k] = v.detach().numpy()
    for k, v in new_params.items():
        if len(v.shape) == 2 and k.endswith(
                ".weight") and k not in bn_w_name_list:
            new_params[k] = v.T
    paddle.save(new_params,
                params_path.replace(".pth", ".pdiparams").replace(
                    ".pt", ".pdiparams").replace(".ckpt", ".pdiparams"))


def main(args):
    project_path = args.project_dir
    save_path = args.save_dir
    params_path = args.pretrain_model
    if params_path is not None:
        params_path = osp.abspath(params_path)
        if osp.isdir(params_path):
            for file in os.listdir(params_path):
                if osp.splitext(file)[-1] in [".pth", ".pt", ".ckpt"]:
                    convert_params(osp.join(params_path, file))
        else:
            convert_params(params_path)
    project_path = osp.abspath(project_path)
    file_dependencies = dict()
    sys.path.append(project_path)
    generate_dependencies(project_path, file_dependencies)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    convert_code(project_path, save_path, file_dependencies)
