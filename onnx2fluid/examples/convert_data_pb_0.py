#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:50:03 2019

@author: Macrobull
"""

# import os, sys
import os
import sys
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper

from collections import OrderedDict as Dict
from glob import glob


def _make_var_name(name):
    """
    make a valid variable name in Python code
    """

    if name == '':
        return '_'
    if name[0].isdigit():
        return 'var_' + name
    for s in ' *?\\/-:':
        name = name.replace(s, '_')
    if name.startswith('_'):
        name = 'var' + name
    return name


data_dir = os.path.dirname(sys.argv[1])
input_names = sys.argv[2].split(':')
output_name = sys.argv[3].split(':')

# Load inputs
inputs = []
for fn in glob(os.path.join(data_dir, 'input_*.pb')):
    tensor = onnx.TensorProto()
    with open(fn, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

# Load outputs
outputs = []
for fn in glob(os.path.join(data_dir, 'output_*.pb')):
    tensor = onnx.TensorProto()
    with open(fn, 'rb') as f:
        tensor.ParseFromString(f.read())
    outputs.append(numpy_helper.to_array(tensor))

inputs = Dict(zip(map(_make_var_name, input_names), inputs))
outputs = Dict(zip(map(_make_var_name, output_name), outputs))

np.savez(data_dir, inputs=inputs, outputs=outputs)
