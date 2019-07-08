#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:22:46 2019

@author: Macrobull
"""

from __future__ import division

import logging
import numpy as np
import torch

from collections import OrderedDict
from typing import (
    TypeVar,
    Any,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Text,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

__all__ = [
    'export_data',
    'export_onnx_with_validation',
]

my_dict = OrderedDict

KT = TypeVar('KT')
VT = TypeVar('VT')


class MyDict(my_dict, Generic[KT, VT]):
    pass


def ensure_list(obj: Union[object, Sequence[object]]) -> List[object]:
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return [obj]


def ensure_tuple(obj: Union[object, Sequence[object]]) -> Tuple[object, ...]:
    if isinstance(obj, (tuple, list, set)):
        return tuple(obj)
    return (obj, )


def flatten_list(obj: List[Union[object, List[object]]],
                 out: Optional[List[object]] = None) -> List[object]:
    assert isinstance(obj, list), 'list type required'

    if out is None:
        out = type(obj)()
    for item in obj:
        if isinstance(item, list):
            flatten_list(item, out)
        else:
            out.append(item)
    return out


def export_data(state_dict: Mapping[Text, Any], prefix: Text = '') -> None:
    """
    export binary data with meta text for raw C++ inference engines
    """

    def str_(obj: object) -> Text:
        if isinstance(obj, (tuple, list, set)):
            return str(obj)[1:-1].replace(' ', '')
        return str(obj)

    prefix_ = prefix + ('_' if prefix else '')
    fp = open('{}.txt'.format(prefix or 'meta'), mode='w')
    for key, value in state_dict.items():
        data = None
        if torch.is_tensor(value):
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


def export_onnx_with_validation(
        model: torch.nn.Module,  # or JITScriptModule
        inputs: Sequence[Union[torch.Tensor, Sequence[object]]],
        export_basepath: Text,
        input_names: Optional[List[Text]] = None,
        output_names: Optional[List[Text]] = None,
        use_npz: bool = True,
        *args,
        **kwargs) -> Sequence[Union[torch.Tensor, Sequence[object]]]:
    """
    export PyTorch model to ONNX model and export sample inputs and outputs in a Numpy file
    """

    is_tuple_or_list = lambda x: isinstance(x, (tuple, list))

    def tensors_to_arrays(tensors: Union[torch.Tensor, Iterable[
            Union[torch.Tensor, Iterable[Any]]]], ) -> List[np.ndarray]:
        if torch.is_tensor(tensors):
            return tensors.data.cpu().numpy()
        return list(map(tensors_to_arrays, tensors))

    def zip_dict(
            keys: Optional[Iterable[Any]],
            values: Sequence[Union[Any, Sequence[Any]]],
    ) -> MyDict[Text, Union[object, MyDict[Text, object]]]:
        keys = keys or range(len(values))
        ret = my_dict()
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
                                input_names=(None if input_names is None else
                                             flatten_list(input_names)),
                                output_names=(None if output_names is None else
                                              flatten_list(output_names)),
                                *args,
                                **kwargs)
    if outputs is None:  # WORKAROUND: for torch.onnx
        training = kwargs.get('training', False)
        with torch.onnx.set_training(model, training):
            outputs = model(*inputs)
    torch_outputs = ensure_tuple(outputs)

    inputs = zip_dict(input_names, tensors_to_arrays(torch_inputs))
    outputs = zip_dict(output_names, tensors_to_arrays(torch_outputs))
    if use_npz:
        np.savez(
            export_basepath + '.npz',
            inputs=inputs,
            outputs=outputs,
        )
    else:
        np.save(export_basepath + '.npy',
                np.asarray(my_dict(inputs=inputs, outputs=outputs)),
                allow_pickle=True)

    return torch_outputs


if __name__ == '__main__':
    from torchvision.models import resnet18 as net

    model = net()
    xb = torch.rand((1, 3, 224, 224))
    export_onnx_with_validation(
        model,
        (xb, ),
        '/tmp/export',
        input_names=[
            'image',
        ],
        output_names=[
            'prob',
        ],
        use_npz=True,
    )
