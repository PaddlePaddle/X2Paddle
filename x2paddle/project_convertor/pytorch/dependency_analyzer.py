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
import sys
import os.path as osp
import ast
import astor
from x2paddle.project_convertor.pytorch.mapper import *
import copy
from .utils import get_dep_file_path


class PtDepInfo:
    """
    PyTorch依赖包信息。
    FROM代表from信息的字符串，例如：torch；
    IMPORT代表import信息系的字符串，例如：nn.functional；
    AS代表as信息的字符串，例如：F；
    DEPENDENCY代表由FROM、IMPORT、AS三者组成的字符串，例如：from torch import nn.functional as F。
    """
    FROM = None
    IMPORT = None
    AS = None
    DEPENDENCY = None


class DependencyAnalyzer(ast.NodeVisitor):
    """ 获取python文件的依赖信息。
        依赖信息由4部分组成：（1）import相关的依赖；（2）赋值；（3）函数；（4）类。

    Args:
        py_file_path (str): python文件的绝对值路径。
        file_dependencies (dict): 当前已经统计的依赖信息，key为python文件的绝对值路径，
                                value为key文件所对应的依赖信息组成的list。
    """

    def __init__(self, py_file_path, file_dependencies):
        self.py_file_path = py_file_path
        self.file_dependencies = file_dependencies
        self.root = ast.parse(open(py_file_path, "rb").read())
        self.scopes_and_dependencies = list()  # 作用域和依赖组成的stack
        self.file_dependencies[self.py_file_path] = list()

    def _get_scope_node(self):
        """ 获取当前节点的作用域。
        """
        scope_node = None
        for i in range(len(self.scopes_and_dependencies)):
            i = -(i + 1)
            sd = self.scopes_and_dependencies[i]
            if not isinstance(sd, PtDepInfo) and not isinstance(sd, ast.Assign):
                scope_node = sd
                break
        return scope_node

    def run(self):
        self.scopes_and_dependencies.append(self.root)
        self.visit(self.root)

    def visit(self, node):
        out = super(DependencyAnalyzer, self).visit(node)

    def visit_ImportFrom(self, node):
        """ 遍历子节点。
        """
        son_nodes = node.names
        for son_node in son_nodes:
            self.visit_alias(son_node, node.module, node.level)

    def visit_Import(self, node):
        """ 遍历子节点。
        """
        son_nodes = getattr(node, "names")
        for son_node in son_nodes:
            self.visit_alias(son_node)

    def visit_alias(self, node, from_name=None, from_level=None):
        """ 构建PtDepInfo并将当前的PtDepInfo放入scopes_and_dependencies。
            如果import字符串为“*”，获取依赖包所在文件的依赖信息加入当前的dependencies；
            反之，直接在dependencies中加入PtDepInfo。
        """
        dep_info = PtDepInfo()
        dep_info.FROM = from_name
        dep_info.IMPORT = getattr(node, "name")
        dep_info.AS = getattr(node, "asname", None)
        if dep_info.IMPORT == "*":
            import_file_path = get_dep_file_path(self.py_file_path, from_level,
                                                 dep_info.FROM)
            if import_file_path not in self.file_dependencies:
                analyzer = DependencyAnalyzer(import_file_path,
                                              self.file_dependencies)
                analyzer.run()
            self.file_dependencies[self.py_file_path].extend(
                self.file_dependencies[import_file_path])
        else:
            dependency_str_list = list()
            if dep_info.FROM is None and from_level is not None:
                dependency_str_list.append("." * from_level)
            elif dep_info.FROM is not None:
                dependency_str_list.append(dep_info.FROM)
            dependency_str_list.append(dep_info.IMPORT)
            dep_info.DEPENDENCY = ".".join(dependency_str_list)
            self.file_dependencies[self.py_file_path].append(dep_info)
        self.scopes_and_dependencies.append(dep_info)

    def visit_FunctionDef(self, node):
        """ 当作用域为ast的根节点时，把函数名放入dependencies。
        """
        if isinstance(self._get_scope_node(), ast.Module):
            self.scopes_and_dependencies.append(node)
            self.file_dependencies[self.py_file_path].append(node.name)

    def visit_ClassDef(self, node):
        """ 当作用域为ast的根节点时，把类名放入dependencies。
        """
        if isinstance(self._get_scope_node(), ast.Module):
            self.scopes_and_dependencies.append(node)
            self.file_dependencies[self.py_file_path].append(node.name)

    def visit_Assign(self, node):
        """ 当作用域为ast的根节点时，把赋值名放入dependencies。
        """
        if isinstance(self._get_scope_node(), ast.Module):
            self.scopes_and_dependencies.append(node)
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    for ele in target.elts:
                        self.file_dependencies[self.py_file_path].append(ele.id)
                elif isinstance(target, ast.Name):
                    self.file_dependencies[self.py_file_path].append(target.id)


def analyze(py_file_path, file_dependencies):
    analyzer = DependencyAnalyzer(py_file_path, file_dependencies)
    analyzer.run()
