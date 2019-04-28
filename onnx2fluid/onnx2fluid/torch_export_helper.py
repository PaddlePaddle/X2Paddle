#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:22:46 2019

@author: Macrobull
"""

import numpy as np
import torch

from collections import OrderedDict as Dict


def ensure_list(obj):
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return [obj]


def ensure_tuple(obj):
    if isinstance(obj, (tuple, list, set)):
        return tuple(obj)
    return (obj, )


def flatten_list(obj, out=None):
    assert isinstance(obj, list)
    if out is None:
        out = type(obj)()
    for item in obj:
        if isinstance(item, list):
            flatten_list(item, out)
        else:
            out.append(item)
    return out


def export_data(state_dict, prefix=''):
    """
	export binary data with meta text for raw C++ inference engines
	"""

    def str_(obj):
        if isinstance(obj, (tuple, list)):
            return str(obj)[1:-1].replace(' ', '')
        return str(obj)

    prefix_ = prefix + ('_' if prefix else '')
    fp = open('{}.txt'.format(prefix if prefix else 'meta'), 'w')
    for key, value in state_dict.items():
        data = None
        if torch and torch.is_tensor(value):
            data = value.data.cpu().numpy()
        elif isinstance(value, np.ndarray):
            data = value
        if data is not None:
            data.tofile('{}{}.bin'.format(prefix_, key))
            fp.write('{}.dtype={}\n'.format(key, str_(data.dtype.name)))
            fp.write('{}.shape={}\n'.format(key, str_(data.shape)))
        else:
            fp.write('{}={}\n'.format(key, str_(value)))
    fp.close()


def export_onnx_with_validation(model,
                                inputs,
                                export_basepath,
                                input_names=None,
                                output_names=None,
                                use_npz=True,
                                *args,
                                **kwargs):
    """
	export PyTorch model to ONNX model and export sample inputs and outputs in a Numpy file
	"""

    is_tuple_or_list = lambda x: isinstance(x, (tuple, list))

    def tensors_to_arrays(tensors):
        if torch.is_tensor(tensors):
            return tensors.data.cpu().numpy()
        arrays = []
        for tensor in tensors:
            arrays.append(tensors_to_arrays(tensor))
        return arrays

    def zip_dict(keys, values):
        ret = Dict()
        for idx, (key, value) in enumerate(zip(keys, values)):
            is_key_list = is_tuple_or_list(key)
            is_value_list = is_tuple_or_list(value)
            assert is_key_list == is_value_list, 'keys and values mismatch'
            if is_value_list:
                ret[str(idx)] = zip_dict(key, value)
            else:
                ret[key] = value
        return ret

    torch_inputs = ensure_tuple(inputs)  # WORKAROUND: for torch.onnx
    outputs = torch.onnx.export(model,
                                torch_inputs,
                                export_basepath + '.onnx',
                                input_names=flatten_list(input_names),
                                output_names=flatten_list(output_names),
                                *args,
                                **kwargs)
    if outputs is None:  # WORKAROUND: for torch.onnx
        outputs = model(*inputs)
    torch_outputs = ensure_tuple(outputs)

    inputs = zip_dict(input_names, tensors_to_arrays(torch_inputs))
    outputs = zip_dict(output_names, tensors_to_arrays(torch_outputs))
    if use_npz:
        np.savez(export_basepath + '.npz', inputs=inputs, outputs=outputs)
    else:
        np.save(export_basepath + '.npy',
                np.array(Dict(inputs=inputs, outputs=outputs)))
    return torch_outputs
