from .utils import *

class ClassConv2D(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "bias", "bias_attr")
    
    def run(self):
        same_attr_count = 8
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    

class ClassConv2DConv2DTranspose(ClassConv2D):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def run(self):
        same_attr_count = 7
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
    
class ClassLinear(ClassConv2D):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
    
class ClassBatchNorm(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "num_features", "num_channels")
        rename_key(self.kwargs, "eps", "epsilon")
        if "momentum" in self.kwargs:
            if isinstance(self.kwargs["momentum"], float):
                self.kwargs["momentum"] = 1 - self.kwargs["momentum"]
            else:
                self.kwargs["momentum"] = "1 - {}".format(self.kwargs["momentum"])
        if "affine" in self.kwargs and not self.kwargs["affine"]:         
            for key in ["weight_attr", "bias_attr"]:
                self.kwargs[key] = "paddle.ParamAttr(learning_rate=0.0)"
#         if "track_running_stats" in self.kwargs:  
#             if isinstance(self.kwargs["track_running_stats"], bool):
#                 self.kwargs["use_global_stats"] = not self.kwargs["track_running_stats"]
#             else:
#                 self.kwargs["use_global_stats"] = "not {}".format(self.kwargs["use_global_stats"])
    
    def delete_attrs(self):
        delete_key(self.kwargs, "affine")
        delete_key(self.kwargs, "track_running_stats")
    
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
    
class ClassMaxPool2D(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "return_indices", "return_mask")
    
    def check_attrs(self):
        assert "dilation" not in self.kwargs, "The dilation is not supported yet in MaxPool2D!"
    
    def run(self):
        same_attr_count = 3
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
    
class ClassReLU(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)   
    
    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")
        
class FuncRelu(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name) 
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
    
    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")
    
        
class FuncDropout(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name) 
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
    
    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")
        
        
class ClassDropout(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name) 

    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")
        
        
class ClassEmbedding(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name) 
        
    def delete_attrs(self):
        delete_key(self.kwargs, "max_norm")
        delete_key(self.kwargs, "norm_type")
        delete_key(self.kwargs, "scale_grad_by_freq")
        
    def check_attrs(self):
        assert "max_norm" not in self.kwargs or self.kwargs["max_norm"] is None, "The max_norm is not supported yet in Embedding!"
        assert "scale_grad_by_freq" not in self.kwargs or not self.kwargs["scale_grad_by_freq"], "The scale_grad_by_freq must be False in Embedding!"
        
    def run(self):
        same_attr_count = 3
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()


class ClassLoss(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)   

    def delete_attrs(self):
        delete_key(self.kwargs, "size_average")
        delete_key(self.kwargs, "reduce")
        
    def process_attrs(self):
        rename_key(self.kwargs, "target", "label")
    
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class ClassCrossEntropyLoss(ClassLoss):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
class ClassBCEWithLogitsLoss(ClassLoss):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
        
class FuncPad(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
    
    
class FuncCrossEntropy(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "target", "label")

class FuncSigmoid(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        
class FuncBinaryCrossEntropyLogits(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "logit", "label")
        rename_key(self.kwargs, "target", "label")
        if not self.kwargs["size_average"]:
            self.kwargs["reduction"] = "sum"
    
    def delete_attrs(self):
        delete_key(self.kwargs, "size_average")
        delete_key(self.kwargs, "reduce")
        
class FuncSoftmax(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "dim", "axis")

    def delete_attrs(self):
        delete_key(self.kwargs, "_stacklevel")
    
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
    
class ClassSoftmax(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "dim", "axis")
        
class FuncLogSoftmax(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "dim", "axis")
        rename_key(self.kwargs, "input", "x")
        
class FuncSmoothL1Loss(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "beta", "delta")
        rename_key(self.kwargs, "target", "label")
        
class FuncAvgPool2d(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        if "count_include_pad" in self.kwargs:
            if isinstance(self.kwargs["count_include_pad"], bool):
                self.kwargs["exclusive"] = not self.kwargs["count_include_pad"]
            else:
                self.kwargs["exclusive"] = "not {}".format(self.kwargs["count_include_pad"])
                
    def delete_attrs(self):
        delete_key(self.kwargs, "count_include_pad")
    
    def run(self):
        same_attr_count = 5
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        return super().run()
        
    
class FuncXavierUniform(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)  
        
    def run(self):
        if len(self.args) >= 1:
            param_name = self.args[0]
            if len(self.args) > 1:
                gain = self.args[1]
            else:
                gain = self.kwargs["gain"] if "gain" in self.kwargs else 1.0
        else:
            param_name = self.kwargs["tensor"]
            gain = self.kwargs["gain"] if "gain" in self.kwargs else 1.0
        code = "{} = {}(gain={})".format(param_name,
                                         self.func_name,
                                         gain)
        return [], code, []