from .register import get_registered_layers

#custom layer import begins
from . import roipooling
from . import priorbox
from . import permute
from . import detectionoutput
from . import normalize
from . import select
from . import shufflechannel
from . import convolutiondepthwise
from . import axpy
from . import upsample
from . import relu6
#custom layer import ends

custom_layers = get_registered_layers()


def set_args(f, params):
    """ set args for function 'f' using the parameters in node.layer.param
    Args:
        f (function): a python function object
        params (object): a object contains attributes needed by f's arguments
    Returns:
        arg_names (list): a list of argument names
        kwargs (dict): a dict contains needed arguments
    """
    argc = f.__code__.co_argcount
    arg_list = f.__code__.co_varnames[0:argc]
    kwargs = {}
    for arg_name in arg_list:
        if hasattr(params, arg_name) and params is not None:
            kwargs[arg_name] = getattr(params, arg_name)
    return arg_list, kwargs


def has_layer(layer_type):
    """ test whether this layer exists in custom layer
    """
    return layer_type in custom_layers


def get_params(layer, layer_type):
    import re
    if layer_type.lower() == "deconvolution" or layer_type.lower(
    ) == "convolutiondepthwise":
        param_name = '_'.join(('convolution', 'param'))
    elif layer_type.lower() == "normalize":
        param_name = '_'.join(('norm', 'param'))
    elif len(layer_type) - len(re.sub("[A-Z]", "", layer_type)) >= 2:
        s = ''
        tmp_name = ''
        for i, ch in enumerate(layer_type):
            if i == 0:
                s += ch.lower()
                continue
            elif ch.isupper() and layer_type[i - 1].islower():
                tmp_name += (s + '_')
                s = ''
            s += ch.lower()
        tmp_name += s
        param_name = '_'.join((tmp_name, 'param'))
    else:
        param_name = '_'.join((layer_type.lower(), 'param'))
    return getattr(layer, param_name, None)


def compute_output_shape(node):
    """ compute the output shape of custom layer
    """
    layer_type = node.layer_type
    assert layer_type in custom_layers, "layer[%s] not exist in custom layers" % (
        layer_type)
    shape_func = custom_layers[layer_type]['shape']
    layer = node.layer
    params = get_params(layer, layer_type)
    arg_names, kwargs = set_args(shape_func, params)
    input_shape = node.input_shape
    return shape_func(input_shape, **kwargs)


def make_custom_layer(node):
    """ get the code which implement the custom layer function
    """
    layer_type = node.layer_type
    assert layer_type in custom_layers, "layer[%s] not exist in custom layers" % (
        layer_type)
    layer_func = custom_layers[layer_type]['layer']
    import inspect
    return inspect.getsource(layer_func), layer_func


def deal_weights(node, data=None):
    """ deal the weights of the custom layer
    """
    layer_type = node.layer_type
    weights_func = custom_layers[layer_type]['weights']
    name = node.layer_name
    return weights_func(name, data)
