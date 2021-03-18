from .utils import *

class ClassDataLoader(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        
    def process_attrs(self):
        pass
    
    def delete_attrs(self):
        delete_key(self.kwargs, "pin_memory")
    
    def check_attrs(self):
        assert "sampler" not in self.kwargs or self.kwargs["sampler"] is None, "The sampler in torch.utils.data.DataLoader can not be set!"
    
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        

