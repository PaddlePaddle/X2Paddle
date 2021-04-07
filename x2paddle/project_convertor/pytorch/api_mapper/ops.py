from .utils import *
from x2paddle.utils import *

class FuncSave(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "pickle_module")
    
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.save"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()

class FuncLoad(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def delete_attrs(self):
        delete_key(self.kwargs, "pickle_module")
        delete_key(self.kwargs, "map_location")
    
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.load"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()
        

class FuncSetDevice(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()
        
    def process_attrs(self):
        self.useful_attrs["device"] = self.args[0]
        self.args[0] = self.target_name
    
    def run(self):
        self.process_attrs()
        insert_codes = list()
        insert_codes.append("{} = {}".format(self.target_name, self.useful_attrs["device"]))
        insert_codes.append("{} = {}.replace('cuda', 'gpu')".format(self.target_name, self.target_name))  
        return insert_codes, generate_api_code(self.func_name, self.args, self.kwargs), []
    
    
class ClassDataParallel(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def delete_attrs(self):
        delete_key(self.kwargs, "device_ids")
        delete_key(self.kwargs, "output_device")
        delete_key(self.kwargs, "dim")
        
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.DataParallel"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            return self.convert_to_paddle()
        
        
class FuncUnSqueeze(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "dim", "axis") 
        
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.unsqueeze"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()
        
        
class FuncMath(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        
    def run(self):
        if self.pytorch_api_name == "torch.sqrt":
            if self.rename_func_name("x2paddle.torch2paddle.sqrt"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.abs":
            if self.rename_func_name("x2paddle.torch2paddle.abs"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.log":
            if self.rename_func_name("x2paddle.torch2paddle.log"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.exp":
            if self.rename_func_name("x2paddle.torch2paddle.exp"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.clip":
            if self.rename_func_name("x2paddle.torch2paddle.clip"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        return self.convert_to_paddle()
        
class FuncArange(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")
        
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.arange"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(3)
            return self.convert_to_paddle()
    
class FuncMatmul(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.matmul"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs()
            return self.convert_to_paddle()
        
        
class FuncCreateParam(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "data", "value")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "requires_grad")
        
    def check_attrs(self):
        assert "requires_grad" not in self.kwargs or self.kwargs["requires_grad"], "The requires_grad must be True in Parameter!"
        
    def run(self):
        if self.rename_func_name(self.func_name):
            if "*" in self.args[0] and "**" not in self.args[0]:
                param_name = self.args[0][1:]
            elif "**" in self.args[0]:
                param_name = "{}['data']".format(self.args[0][2:])
            elif "*" not in self.args[0] and "**" in self.args[1]:
                param_name = self.args[0]
        else:
            self.check_attrs()
            self.process_attrs()
            self.delete_attrs()
            if len(self.args) == 1:
                param_name = self.args[0]
            else:
                param_name = self.kwargs["value"]
        code = "paddle.create_parameter(shape={}.shape, dtype={}.dtype, default_initializer = {}({}))".format(
            param_name, param_name, self.func_name, param_name)
        return [], code, []      
    
    
class FuncNoGrad(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def delete_attrs(self):
        self.args.clear()
        self.kwargs.clear()
        
    
class FuncLogical(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")  
        
    def run(self):
        if self.pytorch_api_name == "torch.bitwise_or":
            if self.rename_func_name("x2paddle.torch2paddle.logical_or"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.bitwise_and":
            if self.rename_func_name("x2paddle.torch2paddle.logical_and"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.bitwise_xor":
            if self.rename_func_name("x2paddle.torch2paddle.logical_xor"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.bitwise_not":
            if self.rename_func_name("x2paddle.torch2paddle.logical_not"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        return self.convert_to_paddle()
        
        
class FuncStack(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def process_attrs(self):
        rename_key(self.kwargs, "tensors", "x")
        rename_key(self.kwargs, "dim", "axis")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out") 
        
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.stack"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            return self.convert_to_paddle()
    
    
class FuncRandperm(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")
        delete_key(self.kwargs, "pin_memory")
        
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.randperm"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            return self.convert_to_paddle()
        
    
class ClassFloatTensor(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def run(self):
        insert_code = "{} = paddle.cast({}, dtype='float32')".format(self.target_name, self.target_name)
        return [], generate_api_code(self.func_name, self.args, self.kwargs), [insert_code]
    
    
class FunTensorBuilder(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()
        
    def process_attrs(self):
        rename_key(self.kwargs, "size", "shape")
        self.useful_attrs["requires_grad"] = self.kwargs["requires_grad"] if "requires_grad" in self.kwargs else False
                
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")
    
    def run(self):
        if self.pytorch_api_name == "torch.full":
            if self.rename_func_name("x2paddle.torch2paddle.full"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.zeros":
            if self.rename_func_name("x2paddle.torch2paddle.zeros"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.ones":
            if self.rename_func_name("x2paddle.torch2paddle.ones"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        out1, out2, out3 = self.convert_to_paddle()
        if isinstance(self.useful_attrs["requires_grad"], str) or not self.useful_attrs["requires_grad"]:
            out2 = "{}.requires_grad_({})".format(out2, self.useful_attrs["requires_grad"])
        return out1, out2, out3
    
        
class FunTensorLike(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        self.useful_attrs["requires_grad"] = self.kwargs["requires_grad"] if "requires_grad" in self.kwargs else False
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")
        
    def run(self):
        if self.pytorch_api_name == "torch.full_like":
            if self.rename_func_name("x2paddle.torch2paddle.full_like"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.zeros_like":
            if self.rename_func_name("x2paddle.torch2paddle.zeros_like"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        if self.pytorch_api_name == "torch.ones_like":
            if self.rename_func_name("x2paddle.torch2paddle.ones_like"):
                return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        out1, out2, out3 = self.convert_to_paddle()
        if isinstance(self.useful_attrs["requires_grad"], str) or not self.useful_attrs["requires_grad"]:
            out2 = "{}.requires_grad_({})".format(out2, self.useful_attrs["requires_grad"])
        return out1, out2, out3