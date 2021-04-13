from .utils import *

class ImageFolderMapper(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def check_attrs(self):
        assert "target_transform" not in self.kwargs, "The target_transform is not supported yet in ImageFolder!"
    
    def run(self):
        if self.pytorch_api_name == "torchvision.datasets.ImageFolder" and \
        self.rename_func_name("x2paddle.torch2paddle.ImageFolder"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_paddle()