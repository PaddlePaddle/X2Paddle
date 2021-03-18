from .utils import *

class ClassAdam(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "params", "parameters")
        rename_key(self.kwargs, "lr", "learning_rate")
        rename_key(self.kwargs, "eps", "epsilon")
        if "betas" in self.kwargs:
            betas = self.kwargs.pop("betas")
            if isinstance(betas, list):
                self.kwargs["beta1"] = betas[0]
                self.kwargs["beta2"] = betas[1]
            else:
                self.kwargs["beta1"], self.kwargs["beta2"] = betas.replace("[", "").replace("]", "").split(",")
                self.kwargs["beta1"] = self.kwargs["beta1"].strip()
                self.kwargs["beta2"] = self.kwargs["beta2"].strip()
    
    def check_attrs(self):
        assert "amsgrad" not in self.kwargs or not self.kwargs["amsgrad"], "The amsgrad in torch.optim.Adam must be False!"
    
    def run(self):
        same_attr_count = 0
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        out1, out2, out3 = super().run()
        return out1, out2, out3
    
class ClassMomentum(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)    

    def process_attrs(self):
        rename_key(self.kwargs, "params", "parameters")
        rename_key(self.kwargs, "lr", "learning_rate")
        rename_key(self.kwargs, "nesterov", "use_nesterov")
        
    def del_attrs(self):
        delete_key(self.kwargs, "dampening")
    
    def check_attrs(self):
        assert "dampening" not in self.kwargs or self.kwargs["dampening"] == 0, "The amsgrad in torch.optim.Adam must be False!"
    
    def run(self):
        same_attr_count = 0
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    