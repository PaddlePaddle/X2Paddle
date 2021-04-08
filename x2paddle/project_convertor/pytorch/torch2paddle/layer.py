import paddle
from functools import partial

def add_layer_function(func):
    setattr(paddle.nn.Layer, func.__name__, func)
    
@property
def module(self):
    if hasattr(self, "_layers"):
        return self._layers
    else:
        return self
setattr(paddle.nn.Layer, "module", module)

@add_layer_function
def load_state_dict(self, state_dict, strict=True):
    for key, param in self.state_dict().items():
        state = state_dict.get(key, None) 
        if state is None:
            if key.endswith(".scale"):
                state_dict[key] = state_dict.pop(key[0: -5] + "weight")
    self.set_state_dict(state_dict)

@add_layer_function
def to(self, *args, **kwargs):
    # TODO(syf): for dtype
    return self

@add_layer_function
def cuda(self):
    return self

@add_layer_function
def apply(self, func):
    func(self)
    
train_tmp = partial(paddle.nn.Layer.train)
@add_layer_function
def train(self, mode=True):
    if mode:
        return train_tmp(self)
    else:
        return paddle.nn.Layer.eval(self)