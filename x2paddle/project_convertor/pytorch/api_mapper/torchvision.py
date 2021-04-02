from .utils import *

class ClassImageFolder(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def check_attrs(self):
        assert "target_transform" not in self.kwargs, "The target_transform is not supported yet in ImageFolder!"
    
    def run(self):
        if self.pytorch_api_name == "torchvision.datasets.ImageFolder" and \
        self.args_has_star("x2paddle.torch2paddle.ImageFolder"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            same_attr_count = 1
            if len(self.args) > same_attr_count:
                new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
                self.kwargs.update(new_kwargs)
                self.args = self.args[:same_attr_count]
            return self.convert_to_paddle()