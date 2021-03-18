from .utils import *

class ClassImageFolder(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def check_attrs(self):
        assert "target_transform" not in self.kwargs, "The target_transform is not supported yet in ImageFolder!"
    
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class ClassNormalize(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")
    
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()