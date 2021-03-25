from .utils import *

class ClassReduceOnPlateau(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()
        
    def process_attrs(self):
        self.kwargs["learning_rate"] = 0.01
     
    def delete_attrs(self):
        self.useful_attrs["optimizer"] = self.kwargs.pop("optimizer")
    
    def check_attrs(self):
        assert "amsgrad" not in self.kwargs, "The amsgrad in torch.optim.Adam must be False!"
    
    def run(self):
        same_attr_count = 0
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        insert_code = "{}._learning_rate = {}".format(self.useful_attrs["optimizer"], self.target_name)
        return [], generate_api_code(self.func_name, self.args, self.kwargs), [insert_code]
    
class ClassCosineAnnealingDecay(ClassReduceOnPlateau):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def check_attrs(self):
        pass
    
class ClassMultiStepDecay(ClassCosineAnnealingDecay):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
    

