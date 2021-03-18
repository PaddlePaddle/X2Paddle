from .utils import *

class FuncSave(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        
    def process_attrs(self):
        pass
    
    def delete_attrs(self):
        delete_key(self.kwargs, "pickle_module")
    
    def check_attrs(self):
        pass
    
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
    

class FuncLoad(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        
    def process_attrs(self):
        pass
    
    def delete_attrs(self):
        delete_key(self.kwargs, "pickle_module")
        delete_key(self.kwargs, "map_location")
    
    def check_attrs(self):
        pass
    
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        

class FuncSetDevice(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        self.target_name = target_name
        self.useful_attrs = dict()
        
    def process_attrs(self):
        self.useful_attrs["device"] = self.args[0]
        self.args[0] = self.target_name
    
    def delete_attrs(self):
        pass
    
    def check_attrs(self):
        pass
    
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        insert_codes = list()
        insert_codes.append("{} = {}".format(self.target_name, self.useful_attrs["device"]))
        insert_codes.append("{} = {}.replace('cuda', 'gpu')".format(self.target_name, self.target_name))  
        return insert_codes, generate_api_code(self.func_name, self.args, self.kwargs), []
    
    
class FuncConcat(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
    
    def process_attrs(self):
        rename_key(self.kwargs, "tensors", "x")
        rename_key(self.kwargs, "dim", "axis")
        if len(self.args) > 0 and isinstance(self.args[0], (list, tuple)):
            self.args[0] = "[{}]".format(", ".join(self.args[0]))
        if "x" in self.kwargs and isinstance(self.kwargs["x"], (list, tuple)):
            self.kwargs["x"] = "[{}]".format(", ".join(self.kwargs["x"]))
    
    def delete_attrs(self):
        pass
    
    def check_attrs(self):
        pass
    
    def run(self):
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
    
    
class ClassDataParallel(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def delete_attrs(self):
        delete_key(self.kwargs, "device_ids")
        delete_key(self.kwargs, "output_device")
        delete_key(self.kwargs, "dim")
        
        
class FuncUnSqueeze(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "dim", "axis") 
        
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class FuncSum(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "dim", "axis") 
        
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class FuncMean(FuncSum):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
class FunBuildTensor(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
                
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")
    
    def run(self):
        shape_str = "{}".format(self.args[0].strip()) if len(self.args) == 1 else "[{}]".format(", ".join(self.args))
        self.args.clear()
        self.args.append(shape_str)
        return super().run()
    
class FunSqrt(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        
class FunAbs(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        
class FuncArange(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")
        
    def run(self):
        same_attr_count = 3
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class FuncMatmul(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        
        
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
        
    
class FuncClip(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out") 
        
    def run(self):
        same_attr_count = 3
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class FuncExp(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out") 
        
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class FuncMaxMin(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        if isinstance(target_name, str):
            self.func_name += "imum"
    
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")  
        
    def run(self):
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        
        
class FuncArgMaxMin(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        
        
class FuncStack(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def process_attrs(self):
        rename_key(self.kwargs, "tensors", "x")
        rename_key(self.kwargs, "dim", "axis")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out") 
        
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class FuncLog(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out") 
        
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
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
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class FuncRand(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        size = list()
        is_str = False
        for i in range(len(self.args)):
            ele = self.args.pop(0)
            if isinstance(ele, str):
                ele = ele.strip()
                is_str = True
            size.append(ele)
        if is_str:
            for i, s in enumerate(size):
                if not isinstance(s, str):
                    size[i] = str(s)
            size = "[{}]".format(", ".join(size))
        self.args.append(size)
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")
        delete_key(self.kwargs, "pin_memory")
        
    def run(self):
        return super().run()
    
class ClassFloatTensor(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    
    def run(self):
        insert_code = "{} = paddle.cast({}, dtype='float32')".format(self.target_name, self.target_name)
        return [], generate_api_code(self.func_name, self.args, self.kwargs), [insert_code]