import inspect

def api_args2kwargs(pytorch_api_name, args, same_attr_count):
    """ 将每个OP的args转为kwargs。
    
    Args:
        pytorch_api_name (str): OP的类型名字。
        args (list): 参数列表。
    """
    def get_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
#             if v.default is not inspect.Parameter.empty
        }
    is_func = True
    if "torch.nn" in pytorch_api_name and "functional"not in pytorch_api_name:
        is_func = False
    if pytorch_api_name.startswith("torchvision"):
        import torchvision
        obj = torchvision
    else:
        import torch
        obj = torch
    for i, part in enumerate(pytorch_api_name.split(".")):
        if i == 0:
            continue
        obj = getattr(obj, part)
    if is_func:
        func = obj
    else:
        func = obj.__init__ 
    default_attrs = get_default_args(func)
    new_kwargs = dict()
    for i, (default_k, default_v) in enumerate(default_attrs.items()):
        if i >= same_attr_count and i < len(args):
            new_kwargs[default_k] = args[i]
    return new_kwargs
    
def rename_key(kwargs, old_key, new_key):
    if old_key in kwargs:
        v = kwargs.pop(old_key)
        kwargs[new_key] = v

def delete_key(kwargs, old_key):
    if old_key in kwargs:
        kwargs.pop(old_key)

def generate_api_code(func_name, args, kwargs):
    for i, arg in enumerate(args):
        if not isinstance(args[i], str):
            args[i] = str(args[i])
    args_str = ", ".join(args)
    kwargs_str_list = list()
    for k, v in kwargs.items():
        kwargs_str_list.append("{}={}".format(k, v))
    kwargs_str = ", ".join(kwargs_str_list)
    if len(args_str) > 0:
        code = "{}({}, {})".format(func_name, args_str, kwargs_str)
    else:
        code = "{}({})".format(func_name, kwargs_str)
    return code


class Mapper(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        self.target_name = target_name
        
    def process_attrs(self):
        pass
     
    def delete_attrs(self):
        pass
    
    def check_attrs(self):
        pass
    
    def run(self):
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []