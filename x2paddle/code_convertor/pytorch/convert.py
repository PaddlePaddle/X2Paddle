import ast
import astor
import os
import os.path as osp
import shutil
import inspect

from tree_updation import Convertor
from module_statistics import MuduleStatistics


file2element = dict()

def write_file(path, tree):
    codes = astor.to_source(tree)
    codes = codes.replace("(...)", "...")
    codes = codes.replace(".pth", ".pdiparams").replace(".pt", ".pdiparams")
    f = open(path, "w")
    f.write(codes)
    f.close()
    
def generate_import_info(folder_path):
    for name in os.listdir(folder_path):
        current_path = osp.join(folder_path, name)
        if osp.isfile(current_path) and current_path.endswith(".py"):
            if current_path in file2element:
                break
            statistics = MuduleStatistics(current_path, file2element)
            statistics.run()
        elif osp.isdir(current_path):
            generate_import_info(current_path)

def convert_code(folder_path, new_folder_path):
    for name in os.listdir(folder_path):
        current_path = osp.join(folder_path, name)
        print(current_path)
        new_current_path = osp.join(new_folder_path, name)
        if osp.isfile(current_path) and current_path.endswith(".py"):
            convertor = Convertor(current_path, file2element)
            convertor.run()
            write_file(new_current_path, convertor.root)
        elif osp.isdir(current_path):
            if not osp.exists(new_current_path):
                os.makedirs(new_current_path)
            convert_code(current_path, new_current_path)
        elif osp.isfile(current_path) and current_path.endswith(".pyc"):
            continue
        elif osp.isdir(current_path) and current_path == "__pycache__":
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
            new_params[k.replace(".running_mean", "_mean")] = v.detach().numpy()
        elif k.endswith(".running_var"):
            new_params[k.replace(".running_var", "_variance")] = v.detach().numpy()
            bn_w_name_list.append(k.replace(".running_var", ".weight"))
        else:
            new_params[k] = v.detach().numpy()
    for k, v in new_params.items():
        if len(v.shape) == 2 and k.endswith(".weight") and k not in bn_w_name_list:
            new_params[k] = v.T
    paddle.save(new_params, params_path.replace(".pth", ".pdiparams").replace(".pt", ".pdiparams"))

def main(project_path, new_project_path, params_path=None):
    project_path = osp.abspath(project_path)
    generate_import_info(project_path)
    if not osp.exists(new_project_path):
        os.makedirs(new_project_path)
    convert_code(project_path, new_project_path)
    if params_path is not None:
        params_path = osp.abspath(params_path)
        convert_params(params_path)
