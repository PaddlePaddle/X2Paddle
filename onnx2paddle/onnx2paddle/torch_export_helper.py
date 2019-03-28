#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:22:46 2019

@author: Macrobull
"""

import numpy as np
import torch

from collections import OrderedDict as Dict


def _ensure_list(obj):
    if isinstance(obj, (list, set, tuple)):
        return list(obj)
    return [obj]


def _ensure_tuple(obj):
    if isinstance(obj, (list, set, tuple)):
        return tuple(obj)
    return (obj, )


def _flatten_list(obj,
                 out=None):
    assert isinstance(obj, list)
    if out is None:
        out = type(obj)()
    for item in obj:
        if isinstance(item, list):
            _flatten_list(item, out)
        else:
            out.append(item)
    return out


def export_data(state_dict,
                prefix=''):
    """
    export binary data with meta text for raw C++ inference engines
    """

    def _str(obj):
        if isinstance(obj, (tuple, list)):
            return str(obj)[1:-1].replace(' ', '')
        return str(obj)

    prefix_ = prefix + ('_' if prefix else '')
    fp = open('{}.txt'.format(prefix if prefix else 'meta'), 'w')
    for key, value in state_dict.items():
        data = None
        if torch and torch.is_tensor(value):
            data = value.data.cpu().numpy()
        elif np and isinstance(value, np.ndarray):
            data = value
        if data is not None:
            data.tofile('{}{}.bin'.format(prefix_, key))
            fp.write('{}.dtype={}\n'.format(key, _str(data.dtype.name)))
            fp.write('{}.shape={}\n'.format(key, _str(data.shape)))
        else:
            fp.write('{}={}\n'.format(key, _str(value)))
    fp.close()


def export_onnx_with_validation(model, inputs, export_basepath,
                                input_names=None, output_names=None,
                                use_npz=True,
                                *args, **kwargs):
    """
    export PyTorch model to ONNX model and export sample inputs and outputs in a Numpy file
    """

    is_list_or_tuple = lambda x: isinstance(x, (list, tuple))

    def _tensors_to_arrays(tensors):
        if torch.is_tensor(tensors):
            return tensors.data.cpu().numpy()
        arrays = []
        for tensor in tensors:
            arrays.append(_tensors_to_arrays(tensor))
        return arrays

    def _zip_dict(keys, values):
        ret = Dict()
        for idx, (key, value) in enumerate(zip(keys, values)):
            is_key_list = is_list_or_tuple(key)
            is_value_list = is_list_or_tuple(value)
            assert is_key_list == is_value_list, 'keys and values mismatch'
            if is_value_list:
                ret[str(idx)] = _zip_dict(key, value)
            else:
                ret[key] = value
        return ret

    torch_inputs = _ensure_tuple(inputs) # WORKAROUND: for torch.onnx
    outputs = torch.onnx.export(model, torch_inputs, export_basepath + '.onnx',
                      input_names=_flatten_list(input_names),
                      output_names=_flatten_list(output_names),
                      *args, **kwargs)
    if outputs is None: # WORKAROUND: for torch.onnx
        outputs = model(*inputs)
    torch_outputs = _ensure_tuple(outputs)

    inputs = _zip_dict(input_names, _tensors_to_arrays(torch_inputs))
    outputs = _zip_dict(output_names, _tensors_to_arrays(torch_outputs))
    if use_npz:
        np.savez(export_basepath + '.npz', inputs=inputs, outputs=outputs)
    else:
        np.save(export_basepath + '.npy',
                np.array(Dict(inputs=inputs, outputs=outputs)))
    return torch_outputs
