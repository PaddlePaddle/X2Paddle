from .utils import *
class ToTensor(Mapper):
    def __init__(self, 
                 func_name,
                 pytorch_api_name, 
                 args, kwargs, 
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "device", "place")
    def run(self):
        if self.rename_func_name("paddle.to_tensor"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # 作用：将paddle与pytorch不同的可变参数替换成字典参数，并生成相应代码
            self.convert_args2kwargs()
            return self.convert_to_paddle()