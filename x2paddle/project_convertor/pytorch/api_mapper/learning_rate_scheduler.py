from .utils import *

class ClassLRScheculer(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()
        
    def process_attrs(self):
        self.kwargs["learning_rate"] = 0.01
     
    def delete_attrs(self):
        self.useful_attrs["optimizer"] = self.kwargs.pop("optimizer")
    
    def run(self):
        if self.pytorch_api_name == "torch.optim.lr_scheduler.ReduceLROnPlateau" and \
        self.rename_func_name("x2paddle.torch2paddle.ReduceOnPlateau"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.pytorch_api_name == "torch.optim.lr_scheduler.CosineAnnealingLR" and \
        self.rename_func_name("x2paddle.torch2paddle.CosineAnnealingDecay"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        elif self.pytorch_api_name == "torch.optim.lr_scheduler.MultiStepLR" and \
        self.rename_func_name("x2paddle.torch2paddle.MultiStepDecay"):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            self.convert_args2kwargs()
            self.check_attrs()
            self.process_attrs()
            self.delete_attrs()
            insert_code = "{}._learning_rate = {}".format(self.useful_attrs["optimizer"], self.target_name)
            return [], generate_api_code(self.func_name, self.args, self.kwargs), [insert_code]
    

    

