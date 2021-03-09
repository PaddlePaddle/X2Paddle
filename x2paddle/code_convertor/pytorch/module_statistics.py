import sys
import os.path as osp
import ast
import astor
from x2paddle.code_convertor.pytorch.mapper import *
import copy


class PytorchImportInfo:
    FROM = None
    IMPORT = None
    AS = None
    IMPORT_STR = None    
    
class MuduleStatistics(ast.NodeVisitor):
    def __init__(self, current_path, file2element):
        self.current_path = current_path
        self.file2element = file2element
        self.root = ast.parse(open(current_path, "rb").read())
        self.scope_and_imports = list() # stack 
        self._stack = list()
        self.modules = list()
        
    def _get_scope_node(self):
        scope_node = None
        for i in range(len(self.scope_and_imports)):
            i = - (i + 1)
            import_info = self.scope_and_imports[i]
            if not isinstance(import_info, PytorchImportInfo) and not isinstance(import_info, ast.Assign):
                scope_node = import_info
                break
        return scope_node 
        
    def _get_father_node(self):
        return self._stack[-2]
    
    def _get_current_index(self, scope_node, node):
        current_id = 0
        for i, n in enumerate(scope_node.body):
            if node == n:
                current_id = i
                break
        return current_id
        
    def run(self):
        self.scope_and_imports.append(self.root)
        self.visit(self.root)
        self.file2element[self.current_path] = self.modules                
        
    def visit(self, node):
        self._stack.append(node)
        out = super(MuduleStatistics, self).visit(node)
        self._stack.pop()
        
    def visit_ImportFrom(self, node):
        scope_node = self._get_scope_node()
        current_id = self._get_current_index(scope_node, node)
        scope_node.body.pop(current_id)
        self._from_name = node.module
        son_nodes = node.names
        for son_node in son_nodes:
            copy_node = copy.deepcopy(node)
            copy_node.names = [son_node]
            self.visit_alias(son_node, copy_node)
            scope_node.body.insert(current_id, copy_node)
        self._from_name = None
        
    def visit_Import(self, node):
        self._from_name = None
        son_nodes = getattr(node, "names")
        for son_node in son_nodes:
            self.visit_alias(son_node, node)
            
        
    def visit_alias(self, node, father_node=None):
        import_info = PytorchImportInfo()
        import_info.FROM = self._from_name
        import_info.IMPORT = getattr(node, "name", None)
        import_info.AS = getattr(node, "asname", None)
        import_str_list = list()
        if import_info.FROM is not None:
            import_str_list.append(import_info.FROM)
        if import_info.IMPORT is not None:
            import_str_list.append(import_info.IMPORT)
        import_info.IMPORT_STR = ".".join(import_str_list)
        if import_str_list[-1] == "*":
            import_str_part = import_info.IMPORT_STR.replace(".*", "")
            import_seg = import_str_part.split(".")
            from_str = ".".join(import_seg[0: -1])
            current_abs_path = osp.dirname(self.current_path)
            sys.path.append(osp.dirname(current_abs_path))
            if len(import_seg) == 1:
                from_str = osp.split(current_abs_path)[-1]
            exec("from {} import {}".format(from_str, import_seg[-1]))
            import_file_path = locals()[import_seg[-1]].__file__
            if import_file_path not in self.file2element:
                module_sta = MuduleStatistics(import_file_path, self.file2element)
                module_sta.run()
            self.modules.extend(self.file2element[import_file_path])
        else:
            self.modules.append(import_info)
        self.scope_and_imports.append(import_info)
        
    def visit_FunctionDef(self, node):
        if isinstance(self._get_scope_node(), ast.Module):
            self.scope_and_imports.append(node)
            self.modules.append(node.name)
        
    def visit_ClassDef(self, node):
        if isinstance(self._get_scope_node(), ast.Module):
            self.scope_and_imports.append(node)
            self.modules.append(node.name)
    
    def visit_Assign(self, node):
        if isinstance(self._get_scope_node(), ast.Module):
            self.scope_and_imports.append(node)
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    for ele in target.elts:
                        self.modules.append(ele.id)
                elif isinstance(target, ast.Name):
                    self.modules.append(target.id)

