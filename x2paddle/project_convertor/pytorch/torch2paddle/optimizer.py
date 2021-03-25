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
                 params, 
                 lr=0.001, 
                 momentum=0.0, 
                 dampening=0, 
                 weight_decay=0.0, 
                 nesterov=False):
        assert dampening == 0, "The dampening must be 0 in SGD!"
        parameters_list = update_parameters(params)
        super().__init__(
             learning_rate=lr, 
             momentum=momentum, 
             parameters=parameters_list, 
             use_nesterov=nesterov, 
             weight_decay=weight_decay,
             grad_clip=None, 
             name=None)
        
    def zero_grad(self):
        return self.clear_grad()
        
        
class Adam(Base_Adam):
    def __init__(self, 
                 params, 
                 lr=0.001, 
                 betas=(0.9, 0.999), 
                 eps=1e-08, 
                 weight_decay=0, 
                 amsgrad=False):
        parameters_list = update_parameters(params)
        super().__init__(
            learning_rate=lr,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=epsilon,
            parameters=parameters_list,
            weight_decay=weight_decay,
            grad_clip=None,
            name=None,
            lazy_mode=False)   
        
    def zero_grad(self):
        return self.clear_grad()
        
