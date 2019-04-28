#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:50:03 2019

@author: Macrobull
"""

import sys
import numpy as np

from collections import OrderedDict as Dict


def make_var_name(name):
    """
	make a valid variable name in Python code
	"""

    if name == '':
        return '_'
    if name[0].isdigit():
        return 'var_' + name
    for s in ' \\|/:':  #
        name = name.replace(s, '_')
    if name.startswith('_'):
        name = 'var' + name
    return name


fn = sys.argv[1]
input_names = sys.argv[2].split(',')
output_names = sys.argv[3].split(',')
squeeze_data = len(sys.argv) > 4

data = np.load(fn, encoding='bytes')
input_data = data['inputs']
output_data = data['outputs']

while squeeze_data and input_data.ndim > 4 and input_data.shape[0] == 1:
    input_data = input_data.squeeze(0)
while squeeze_data and output_data.ndim > 2 and output_data.shape[0] == 1:
    output_data = output_data.squeeze(0)

inputs = Dict(zip(map(make_var_name, input_names), [input_data]))
outputs = Dict(zip(map(make_var_name, output_names), [output_data]))

np.savez(fn, inputs=inputs, outputs=outputs)  # overwrite
