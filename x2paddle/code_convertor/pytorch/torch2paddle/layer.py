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
    return train_tmp(self)