import sys
import os.path as osp
import ast
import astor
from x2paddle.project_convertor.pytorch.mapper import *
import copy


class PytorchDependencyInfo:
    """
    依赖包信息。
    FROM代表from信息的字符串，例如：torch；
    IMPORT代表import信息系的字符串，例如：nn.functional；
    AS代表as信息的字符串，例如：F；
    DEPENDENCY代表由FROM、IMPORT、AS三者组成的字符串，例如：from torch import nn.functional as F。
    """
    FROM = None
    IMPORT = None
    AS = None
    DEPENDENCY = None    
    
class DependencyAnalysis(ast.NodeVisitor):
    """ 获取python文件的依赖信息。
        依赖信息由4部分组成：（1）import相关的依赖；（2）赋值；（3）函数；（4）类。
        
    Args:
        py_file_path (str): python文件的绝对值路径。
        file_dependency (dict): 当前已经统计的依赖信息，key为python文件的绝对值路径，
                                value为key文件所对应的依赖信息组成的list。
    """
    def __init__(self, py_file_path, file_dependency):
        self.py_file_path = py_file_path
        self.file_dependency = file_dependency
        self.root = ast.parse(open(py_file_path, "rb").read())
        self.scopes_and_dependencies = list()    # 作用域和依赖组成的stack
        self.nodes = list()    # ast节点组成的stack
        self.dependencies = list()    # 依赖组成的list
        
    def _get_scope_node(self):
        """ 获取当前节点的作用域。
        """
        scope_node = None
        for i in range(len(self.scopes_and_dependencies)):
            i = - (i + 1)
            sd = self.scopes_and_dependencies[i]
            if not isinstance(sd, PytorchDependencyInfo) and not isinstance(sd, ast.Assign):
                scope_node = sd
                break
        return scope_node 
    
    def _get_current_index(self, scope_node, node):
        """ 获取当前节点在其作用域中的索引序号。
        """
        current_id = 0
        for i, n in enumerate(scope_node.body):
            if node == n:
                current_id = i
                break
        return current_id
    
    def _generate_file_path(self, dependency_info):
        """ 根据from信息获取依赖包所在文件。如果from字符串中存在相对路径（出现"."），
            则根据相对路径找到相应的文件；反之，执行import语句找到相应依赖的文件。
        """
        if self._level > 0:
            l = self._level
            path = self.py_file_path
            while l > 0:
                path, folder_or_file = osp.split(path)
                l -= 1
            if dependency_info.FROM is None:
                import_file_path = osp.join(path, "__init__.py")
            else:
                path = osp.join(path, osp.join(*dependency_info.FROM.split(".")))
                if osp.exists(path + ".py"):
                    import_file_path = path + ".py"
                else:
                    import_file_path = osp.join(path, "__init__.py")
        else:
            if len(dependency_info.FROM.split(".")) == 1:
                exec("import {}".format(dependency_info.FROM))
                import_file_path = locals()[dependency_info.FROM].__file__
            else:
                from_seg = dependency_info.FROM.split(".")
                import_str = from_seg[-1]
                from_str = ".".join(from_seg[0: -1])
                exec("from {} import {}".format(from_str, import_str))
                import_file_path = locals()[import_str].__file__
        return import_file_path
        
    def run(self):
        self.scopes_and_dependencies.append(self.root)
        self.visit(self.root)
        self.file_dependency[self.py_file_path] = self.dependencies                
        
    def visit(self, node):
        self.nodes.append(node)
        out = super(DependencyAnalysis, self).visit(node)
        self.nodes.pop()
        
    def visit_ImportFrom(self, node):
        """ 遍历子节点。
        """
        scope_node = self._get_scope_node()
        current_id = self._get_current_index(scope_node, node)
        scope_node.body.pop(current_id)
        self._from_name = node.module
        self._level = node.level
        son_nodes = node.names
        for son_node in son_nodes:
            self.visit_alias(son_node)
        self._from_name = None
        self._level = None
        
    def visit_Import(self, node):
        """ 遍历子节点。
        """
        self._from_name = None
        self._level = None
        son_nodes = getattr(node, "names")
        for son_node in son_nodes:
            self.visit_alias(son_node)
        
    def visit_alias(self, node):
        """ 构建PytorchDependencyInfo并将当前的PytorchDependencyInfo放入scopes_and_dependencies。
            如果import字符串为“*”，获取依赖包所在文件的依赖信息加入当前的dependencies；
            反之，直接在dependencies中加入PytorchDependencyInfo。
        """
        dependency_info = PytorchDependencyInfo()
        dependency_info.FROM = self._from_name
        dependency_info.IMPORT = getattr(node, "name")
        dependency_info.AS = getattr(node, "asname", None)
        if dependency_info.IMPORT == "*":
            import_file_path = self._generate_file_path(dependency_info)
            if import_file_path not in self.file_dependency:
                module_sta = DependencyAnalysis(import_file_path, self.file_dependency)
                module_sta.run()
            self.dependencies.extend(self.file_dependency[import_file_path])
        else:
            dependency_str_list = list()
            if dependency_info.FROM is None and self._level is not None:
                dependency_str_list.append("." * self._level)
            elif dependency_info.FROM is not None:
                dependency_str_list.append(dependency_info.FROM)
            dependency_str_list.append(dependency_info.IMPORT)
            dependency_info.DEPENDENCY = ".".join(dependency_str_list)
            self.dependencies.append(dependency_info)
        self.scopes_and_dependencies.append(dependency_info)
        
    def visit_FunctionDef(self, node):
        """ 当作用域为ast的根节点时，把函数名放入dependencies。
        """
        if isinstance(self._get_scope_node(), ast.Module):
            self.scopes_and_dependencies.append(node)
            self.dependencies.append(node.name)
        
    def visit_ClassDef(self, node):
        """ 当作用域为ast的根节点时，把类名放入dependencies。
        """
        if isinstance(self._get_scope_node(), ast.Module):
            self.scopes_and_dependencies.append(node)
            self.dependencies.append(node.name)
    
    def visit_Assign(self, node):
        """ 当作用域为ast的根节点时，把赋值名放入dependencies。
        """
        if isinstance(self._get_scope_node(), ast.Module):
            self.scopes_and_dependencies.append(node)
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    for ele in target.elts:
                        self.dependencies.append(ele.id)
                elif isinstance(target, ast.Name):
                    self.dependencies.append(target.id)

def run(py_file_path, file_dependency):
    analysis = DependencyAnalysis(py_file_path, file_dependency)
    analysis.run()