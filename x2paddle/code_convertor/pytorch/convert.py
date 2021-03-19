import ast
import astor
import os
import os.path as osp
import shutil
import argparse
from six import text_type as _text_type

from dependency_statistics import run as run_dependency
from ast_update import run as run_ast


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_path",
        "-p",
        type=_text_type,
        default=None,
        help="define project folder path for pytorch")
    parser.add_argument(
        "--save_path",
        "-s",
        type=_text_type,
        default=None,
        help="path to save translated code")
    parser.add_argument(
        "--params_path",
        "-pa",
        type=_text_type,
        default=None,
        help="weight file of pytorch model")
    return parser


def write_file(path, tree):
    codes = astor.to_source(tree)
    codes = codes.replace("(...)", "...")
    f = open(path, "w")
    f.write(codes)
    f.close()
    
def generate_dependency(folder_path, file_dependency):
    for name in os.listdir(folder_path):
        current_path = osp.join(folder_path, name)
        if osp.isfile(current_path) and current_path.endswith(".py"):
            if current_path in file_dependency:
                break
            run_dependency(current_path, file_dependency)
        elif osp.isdir(current_path):
            generate_dependency(current_path, file_dependency)

def convert_code(folder_path, new_folder_path, file_dependency):
    for name in os.listdir(folder_path):
        current_path = osp.join(folder_path, name)
        print(current_path)
        new_current_path = osp.join(new_folder_path, name)
        if osp.isfile(current_path) and current_path.endswith(".py"):
            root = run_ast(current_path, file_dependency)
            write_file(new_current_path, root)
        elif osp.isdir(current_path):
            if not osp.exists(new_current_path):
                os.makedirs(new_current_path)
            convert_code(current_path, new_current_path, file_dependency)
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
    file_dependency = dict()
    generate_dependency(project_path, file_dependency)
    if not osp.exists(new_project_path):
        os.makedirs(new_project_path)
    convert_code(project_path, new_project_path, file_dependency)
    if params_path is not None:
        params_path = osp.abspath(params_path)
        convert_params(params_path)

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    main(args.project_path, args.save_path, args.params_path)        