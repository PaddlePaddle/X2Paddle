from paddle.optimizer import Momentum as Base_Momentum
from paddle.optimizer import Adam as Base_Adam
from paddle.regularizer import L2Decay

def update_parameters(parameters):
    parameters_list = list()
    if parameters is not None:
        for items in parameters:
            if isinstance(items, dict):
                params = items["params"]
                if "lr" in items:
                    for p in params:
                        p.optimize_attr["learning_rate"] = items["lr"] / p.optimize_attr["learning_rate"]
                if "weight_decay" in items:
                    for p in params:
                        if isinstance(items["weight_decay"], float):
                            p.regularizer = L2Decay(items["weight_decay"])
                        else:
                            p.regularizer = weight_decay
                parameters_list.extend(params)
            else:
                parameters_list.append(items)
    return parameters_list
                    

class Momentum(Base_Momentum):
    def __init__(self, 
                 learning_rate=0.001, 
                 momentum=0.9, 
                 parameters=None, 
                 use_nesterov=False, 
                 weight_decay=None,
                 grad_clip=None, 
                 name=None):
        parameters_list = update_parameters(parameters)
        super().__init__(
             learning_rate=learning_rate, 
             momentum=momentum, 
             parameters=parameters_list, 
             use_nesterov=use_nesterov, 
             weight_decay=weight_decay,
             grad_clip=grad_clip, 
             name=name)
        
    def zero_grad(self):
        return self.clear_grad()
        
class Adam(Base_Adam):
    def __init__(self, 
                 learning_rate=0.001, 
                 beta1=0.9, 
                 beta2=0.999, 
                 epsilon=1e-08, 
                 parameters=None, 
                 weight_decay=None, 
                 grad_clip=None, 
                 name=None, 
                 lazy_mode=False):
        parameters_list = update_parameters(parameters)
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters_list,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
            lazy_mode=lazy_mode)   
        
    def zero_grad(self):
        return self.clear_grad()
        
