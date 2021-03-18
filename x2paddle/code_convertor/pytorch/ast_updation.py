import ast
import astor
import sys
from x2paddle.code_convertor.pytorch.mapper import *
import copy
import os.path as osp


class DependencyInfo:
    """
    依赖包信息。
    PYTORCH_FROM代表pytorch from信息的字符串，例如：torch；
    PADDLE_FROM代表paddle from信息的字符串，例如：paddle；
    PYTORCH_IMPORT代表pytorch import信息系的字符串，例如：nn.functional；
    PADDLE_IMPORT代表paddle import信息系的字符串，例如：nn.functional；
    AS代表as信息的字符串，例如：F；
    PYTORCH_DEPENDENCY代表由PYTORCH_FROM、PYTORCH_IMPORT、AS三者组成的字符串，例如：from torch import nn.functional as F。
    PADDLE_DEPENDENCY代表由PADDLE_FROM、PADDLE_IMPORT、AS三者组成的字符串，例如：from paddle import nn.functional as F。
    """
    PYTORCH_FROM = None
    PADDLE_FROM = None
    PYTORCH_IMPORT = None
    PADDLE_IMPORT = None
    AS = None
    PYTORCH_DEPENDENCY = None
    PADDLE_DEPENDENCY = None

    
class AstUpdation(ast.NodeVisitor):
    """ 更新ast树，将ast树中PyTorch相关的节点转为Paddle相关的节点。
        
    Args:
        py_file_path (str): python文件的绝对值路径。
        file_dependency (dict): 当前已经统计的依赖信息，key为python文件的绝对值路径，
                                value为key文件所对应的依赖信息组成的list。
    """
    def __init__(self, py_file_path, file_dependency):
        self.py_file_path = py_file_path
        self.root = ast.parse(open(py_file_path, "rb").read())
        self.file_dependency = file_dependency
        self.scopes_and_dependencies = list()    # 作用域和依赖组成的stack
        self.nodes = list()    # ast节点组成的stack

    def _get_scope_node(self):
        """ 获取当前节点的作用域。
        """
        scope_node = None
        for i in range(len(self.scopes_and_dependencies)):
            i = - (i + 1)
            sd = self.scopes_and_dependencies[i]
            if not isinstance(sd, DependencyInfo) and not isinstance(sd, ast.Assign):
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
    
    def _get_father_node(self):
        """ 获取父节点。
        """
        return self.nodes[-2]
    
    def _get_complete_api(self, api_part_name):
        """ 根据部分api名字获取PyTorch的api全名。
            情况1：依赖是DependencyInfo，但其PADDLE_IMPORT为None（非PyTorch的依赖），则pytorch_api为None。
            情况2：依赖是DependencyInfo，且DependencyInfo的部分PyTorch属性以“torch”开头，则pytorch_api为完整api。
            情况3：依赖是ast.Assign节点，则pytorch_api为None。
        """
        pytorch_api = None
        dependency_info = None
        for i in range(len(self.scopes_and_dependencies)):
            i = - (i + 1)
            dependency_info = self.scopes_and_dependencies[i]
            if isinstance(dependency_info, DependencyInfo):
                if dependency_info.PADDLE_IMPORT is None:
                    continue
                if dependency_info.AS is not None and api_part_name.startswith(dependency_info.AS):
                    if (dependency_info.PYTORCH_FROM is not None and "torch" in dependency_info.PYTORCH_FROM) or \
                            (dependency_info.PYTORCH_IMPORT is not None and "torch" in dependency_info.PYTORCH_IMPORT):
                        if api_part_name.endswith(dependency_info.AS):
                            pytorch_api = api_part_name.replace(dependency_info.AS, dependency_info.PYTORCH_DEPENDENCY)
                        else:
                            pytorch_api = api_part_name.replace(dependency_info.AS + ".", 
                                                                dependency_info.PYTORCH_DEPENDENCY + ".")
                        break
                elif api_part_name.startswith(dependency_info.PYTORCH_IMPORT):
                    if (dependency_info.PYTORCH_FROM is not None and "torch" in dependency_info.PYTORCH_FROM) or \
                            (dependency_info.PYTORCH_IMPORT is not None and "torch" in dependency_info.PYTORCH_IMPORT):
                        if api_part_name.endswith(dependency_info.PYTORCH_IMPORT):
                            pytorch_api = api_part_name.replace(dependency_info.PYTORCH_IMPORT, 
                                                                dependency_info.PYTORCH_DEPENDENCY)
                        else:
                            pytorch_api = api_part_name.replace(dependency_info.PYTORCH_IMPORT + ".", 
                                                                dependency_info.PYTORCH_DEPENDENCY + ".")
                        break 
            elif isinstance(dependency_info, ast.Assign):
                is_customized = False
                for s in astor.to_source(dependency_info.targets[0]).split(","):
                    if api_part_name.split(".")[0] == s.strip():
                        is_customized = True
                        break
                if is_customized:
                    break
        return pytorch_api, dependency_info
    
    def _rename(self, name, dependency_info, pytorch_api, paddle_api):
        """ 对函数名进行重命名。
            例如：将nn.Conv2d替换为nn.Conv2D。
        """
        pytorch_api_seg = pytorch_api.split(dependency_info.PYTORCH_IMPORT)
        if paddle_api.startswith(dependency_info.PADDLE_IMPORT + ".") or \
                paddle_api.endswith("." + dependency_info.PADDLE_IMPORT) or  \
                "." + dependency_info.PADDLE_IMPORT + "." in paddle_api:
            paddle_api_seg = paddle_api.split(dependency_info.PADDLE_IMPORT)
            if dependency_info.AS is None:
                name = name.replace(dependency_info.PYTORCH_IMPORT + pytorch_api_seg[-1], 
                                    dependency_info.PADDLE_IMPORT + paddle_api_seg[-1])
            else:
                name = name.replace(pytorch_api_seg[-1], paddle_api_seg[-1])
        return name
    
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
            if dependency_info.PYTORCH_FROM is None:
                import_file_path = osp.join(path, "__init__.py")
            else:
                path = osp.join(path, osp.join(*dependency_info.PYTORCH_FROM.split(".")))
                if osp.exists(path + ".py"):
                    import_file_path = path + ".py"
                else:
                    import_file_path = osp.join(path, "__init__.py")
        else:
            if len(dependency_info.PYTORCH_FROM.split(".")) == 1:
                exec("import {}".format(dependency_info.PYTORCH_FROM))
                import_file_path = locals()[dependency_info.PYTORCH_FROM].__file__
            else:
                from_seg = dependency_info.PYTORCH_FROM.split(".")
                import_str = from_seg[-1]
                from_str = ".".join(from_seg[0: -1])
                exec("from {} import {}".format(from_str, import_str))
                import_file_path = locals()[import_str].__file__
        return import_file_path
        
    def run(self):
        self.scopes_and_dependencies.append(self.root)
        self.visit(self.root)
        for i, node in enumerate(self.root.body):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                self.root.body.insert(i, ast.parse("from x2paddle import torch2paddle").body[0])
                self.root.body.insert(i, ast.parse("import paddle").body[0])
                break
        
    def visit(self, node):
        self.nodes.append(node)
        out = super(AstUpdation, self).visit(node)
        self.nodes.pop()
        if out is not None:
            return out
        else:
            try:
                return astor.to_source(node)
            except Exception:
                return None
        
    def visit_ImportFrom(self, node):
        """ 1. 遍历子节点。
            2. 将当前from依赖中的多个import拆分成多个import。
                例如：from torch import nn, utils 这个node
                     拆分为：node1：from torch import nn
                            node2：from torch import utils
                拆分原因：
                    在paddle中父依赖包可能不一致。
        """
        scope_node = self._get_scope_node()
        current_id = self._get_current_index(scope_node, node)
        scope_node.body.pop(current_id)
        self._from_name = node.module
        self._level = node.level
        son_nodes = node.names
        for son_node in son_nodes:
            copy_node = copy.deepcopy(node)
            copy_node.names = [son_node]
            self.visit_alias(son_node, copy_node)
            scope_node.body.insert(current_id, copy_node)
        self._from_name = None
        self._level = None
        
    def visit_Import(self, node):
        """ 遍历子节点。
        """
        self._from_name = None
        self._level = None
        son_nodes = getattr(node, "names")
        for son_node in son_nodes:
            self.visit_alias(son_node, node)
            
        
    def visit_alias(self, node, father_node=None):
        """ 构建DependencyInfo并将其放入scopes_and_dependencies。
            如果import字符串为“*”，获取依赖包所在文件的依赖信息并转换为DependencyInfo加入当前的scopes_and_dependencies；
            反之，直接在dependencies中加入DependencyInfo。
        """
        dependency_info = DependencyInfo()
        dependency_info.PYTORCH_FROM = self._from_name
        dependency_info.PYTORCH_IMPORT = getattr(node, "name")
        dependency_info.AS = getattr(node, "asname", None)
        if dependency_info.PYTORCH_IMPORT == "*":
            import_file_path = self._generate_file_path(dependency_info)
            pytorch_dependencies = self.file_dependency[import_file_path]
            for pytorch_dependency_info in pytorch_dependencies:
                current_dependency_info = DependencyInfo()
                if not isinstance(pytorch_dependency_info, str):
                    current_dependency_info.PYTORCH_FROM = pytorch_dependency_info.FROM
                    current_dependency_info.PYTORCH_IMPORT = pytorch_dependency_info.IMPORT
                    current_dependency_info.AS = pytorch_dependency_info.AS
                    current_dependency_info.PYTORCH_DEPENDENCY = pytorch_dependency_info.DEPENDENCY
                    if "torch" in current_dependency_info.PYTORCH_DEPENDENCY:
                        current_dependency_info.PADDLE_DEPENDENCY = \
                                API_MAPPER[current_dependency_info.PYTORCH_DEPENDENCY][0]
                        if current_dependency_info.PYTORCH_FROM is not None:
                            seg = current_dependency_info.PADDLE_DEPENDENCY.split(".")
                            current_dependency_info.PADDLE_IMPORT = seg[-1]
                            current_dependency_info.PADDLE_FROM = \
                                    current_dependency_info.PADDLE_DEPENDENCY.replace("." + seg[-1], "")
                        else:
                            current_dependency_info.PADDLE_IMPORT = current_dependency_info.PADDLE_DEPENDENCY
                else:
                    if isinstance(pytorch_dependency_info, str):
                        current_dependency_info.PADDLE_DEPENDENCY = pytorch_dependency_info
                    else:
                        current_dependency_info.PADDLE_DEPENDENCY = pytorch_dependency_info.PYTORCH_DEPENDENCY
                self.scopes_and_dependencies.append(current_dependency_info)
            return
        dependency_str_list = list()
        if dependency_info.PYTORCH_FROM is None and self._level is not None:
            dependency_str_list.append("." * self._level)
        elif dependency_info.PYTORCH_FROM is not None:
            dependency_str_list.append(dependency_info.PYTORCH_FROM)
        dependency_str_list.append(dependency_info.PYTORCH_IMPORT)
        dependency_info.PYTORCH_DEPENDENCY = ".".join(dependency_str_list)
        if dependency_info.PYTORCH_DEPENDENCY.startswith("torch"):
            dependency_info.PADDLE_DEPENDENCY = API_MAPPER[dependency_info.PYTORCH_DEPENDENCY][0]
            if dependency_info.PYTORCH_FROM is not None:
                seg = dependency_info.PADDLE_DEPENDENCY.split(".")
                setattr(node, "name", seg[-1])
                setattr(father_node, "module", dependency_info.PADDLE_DEPENDENCY.replace("." + seg[-1], ""))
                dependency_info.PADDLE_IMPORT = seg[-1]
                dependency_info.PADDLE_FROM = dependency_info.PADDLE_DEPENDENCY.replace("." + seg[-1], "")
            else:
                setattr(node, "name", dependency_info.PADDLE_DEPENDENCY) 
                dependency_info.PADDLE_IMPORT = dependency_info.PADDLE_DEPENDENCY
        else:
            dependency_info.PADDLE_DEPENDENCY = dependency_info.PYTORCH_DEPENDENCY   
        self.scopes_and_dependencies.append(dependency_info)
        
    def visit_Name(self, node):
        """ 获取字符串名字。
        """
        return getattr(node, "id")
    
    def visit_Attribute(self, node):
        """ 对属性字符串满足以下4种情况时进行替换：
            情况1 —— Class A(nn.Module)：将nn.Module替换为nn.Layer；
            情况2 —— a = (1, 2, nn.Module)：将nn.Module替换为nn.Layer；
            情况3 —— torch.float32：将torch.float32替换为"float32"；
            情况4 —— isinstance(a, nn.Module)：将nn.Module替换为nn.Layer。
        """
        value_node = node.value
        attr = node.attr
        name = self.visit(value_node)
        attr_str = name + "." + attr
        pytorch_api, dependency_info = self._get_complete_api(attr_str)
        if pytorch_api in API_MAPPER:
            paddle_api = API_MAPPER[pytorch_api][0]
            father_node = self._get_father_node()
            if isinstance(father_node, ast.ClassDef):
                attr_str = self._rename(attr_str, dependency_info, pytorch_api, paddle_api)
                if node in father_node.bases:
                    father_node.bases[0] = ast.parse(attr_str).body[0].value
                return attr_str
            elif isinstance(father_node, ast.Tuple):
                for i, elts_node in enumerate(father_node.elts):
                    if astor.to_source(elts_node).strip() == attr_str:
                        father_node.elts[i] = ast.parse(paddle_api).body[0].value
                return paddle_api
            elif isinstance(father_node, ast.FunctionDef):
                father_node.returns = ast.parse(paddle_api).body[0].value
                return paddle_api
            elif isinstance(father_node, ast.arg):
                attr_str = self._rename(attr_str, dependency_info, pytorch_api, paddle_api)
                father_node.annotation = ast.parse(attr_str).body[0].value
                return attr_str
            elif isinstance(father_node, ast.Call) and getattr(father_node.func, "id", None) == "isinstance":
                for i, arg_node in enumerate(father_node.args):
                    if astor.to_source(arg_node).strip() == attr_str:
                        father_node.args[i] = ast.parse(paddle_api).body[0].value
                return paddle_api
            elif not isinstance(father_node, ast.Call):
                # 对torch.float32的处理
                for k, v in father_node.__dict__.items():
                    if v == node and pytorch_api in API_MAPPER:
                        father_node.k = ast.parse(paddle_api).body[0].value
                        break
                return attr_str            
        return attr_str
    
    def visit_Num(self, node):
        """ 返回数值。
        """
        return getattr(node, "n")
    
    def visit_keyword(self, node):
        """ 返回键值对。
            【注意】当value是API_MAPPER中的key时，需要替换为API_MAPPER中对应的Paddle API。
        """
        key = getattr(node, "arg")
        value_node = getattr(node, "value")
        value = self.visit(value_node)
        if value in API_MAPPER:
            value = API_MAPPER[value][0]
        return {key: value}
    
    def visit_Tuple(self, node):
        """ 返回tuple。
        """
        elts_nodes = getattr(node, "elts")
        elts = list()
        for elts_node in elts_nodes:
            elts.append(self.visit(elts_node))
        elts = tuple(elts)
        return elts
    
    def visit_Assign(self, node):
        """ 1. 将Assign节点加入scopes_and_dependencies；
            2. 遍历Assign节点的子节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        """ 1. 获取原始函数名并更新为新的函数名。
            2. 获取args和kwargs。
            3. 根据API_MAPPER映射需要更新的操作，对参数进行处理。
            4. 如果有前缀代码和后缀代码，则需要添加相应节点。
        """
        # 获取函数名
        func_node = node.func
        func_name = self.visit(func_node) 
        pytorch_api, dependency_info = self._get_complete_api(func_name)
        if pytorch_api is None:
            self.generic_visit(node)
            return
        paddle_api = API_MAPPER[pytorch_api][0]
        func_name = self._rename(func_name, dependency_info, pytorch_api, paddle_api)
        setattr(node, "func", ast.parse(func_name).body[0].value)
        
        # 获取args
        args_nodes = getattr(node, "args")
        args_list = list()
        for args_node in args_nodes:
            args_list.append(self.visit(args_node))
                
        # 获取keywords
        keywords_nodes = getattr(node, "keywords")
        kw_dict = dict()
        for keywords_node in keywords_nodes:
            kw_dict.update(self.visit(keywords_node))
            
        if API_MAPPER[pytorch_api][1] is None:
            return
                
        target_name = None
        father_node = self._get_father_node()
        if isinstance(father_node, ast.Assign):
            target_node = father_node.targets[0]
            target_name = self.visit(target_node)
        mapper = API_MAPPER[pytorch_api][1](func_name, pytorch_api, args_list, kw_dict, target_name)
        prefix_insert_codes, new_code, suffix_insert_codes = mapper.run()
        scope_node = self._get_scope_node()
        if isinstance(ast.parse(new_code).body[0], ast.Assign):
            node_index = self._get_current_index(scope_node, node)
            scope_node.body[node_index] = ast.parse(new_code).body[0]
        else:
            new_call_node = ast.parse(new_code).body[0].value
            setattr(node, "func", new_call_node.func) # 修改了fun_name
            setattr(node, "args", new_call_node.args)
            setattr(node, "keywords", new_call_node.keywords)
        for i, n in enumerate(scope_node.body):
            if father_node == n:
                for code in prefix_insert_codes:
                    scope_node.body.insert(i, ast.parse(code).body[0])
                    i += 1 
                break
        for i, n in enumerate(scope_node.body):
            if father_node == n:
                j = i + 1
                for code in suffix_insert_codes:
                    scope_node.body.insert(j, ast.parse(code).body[0])
                    j += 1  
                break
        
    def visit_FunctionDef(self, node):
        """ 1. 将FunctionDef节点加入scopes_and_dependencies；
            2. 遍历FunctionDef节点的子节点；
            3. 去除scopes_and_dependencies中FunctionDef节点以及之后的节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        last_node = self.scopes_and_dependencies.pop(-1)
        while isinstance(last_node, DependencyInfo): 
            last_node = self.scopes_and_dependencies.pop(-1)
        
        
    def visit_ClassDef(self, node):
        """ 1. 将ClassDef节点加入scopes_and_dependencies；
            2. 遍历ClassDef节点的子节点；
            3. 去除scopes_and_dependencies中ClassDef节点以及之后的节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        last_node = self.scopes_and_dependencies.pop(-1)
        while isinstance(last_node, DependencyInfo): 
            last_node = self.scopes_and_dependencies.pop(-1)
            
    def visit_If(self, node):
        """ 1. 将If节点加入scopes_and_dependencies；
            2. 遍历If节点的子节点；
            3. 去除scopes_and_dependencies中If节点以及之后的节点。
        """
        self.scopes_and_dependencies.append(node)
        self.generic_visit(node)
        last_node = self.scopes_and_dependencies.pop(-1)
        while isinstance(last_node, DependencyInfo): 
            last_node = self.scopes_and_dependencies.pop(-1)
            
            
    def visit_ExtSlice(self, node):
        """ 将Index节点替换替换为Num。
        """
        dim_nodes = node.dims
        for i, dim_node in enumerate(dim_nodes):
            if isinstance(dim_node, ast.Index):
                dim_nodes[i] = dim_node.value
            else:
                self.visit(dim_node)
                
    def visit_Str(self, node):
        """ 修改模型参数的后缀名。
        """
        setattr(node, "s", node.s.replace(".pth", ".pdiparams").replace(".pt", ".pdiparams"))

            
def run(py_file_path, file_dependency):
    updation = AstUpdation(py_file_path, file_dependency)
    updation.run()
    return updation.root