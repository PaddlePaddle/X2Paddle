#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:50:03 2019

@author: Macrobull
"""

import sys
import numpy as np

from collections import OrderedDict as Dict

fn = sys.argv[1]
input_names = sys.argv[2].split(':')
output_name = sys.argv[3].split(':')

data = np.load(fn)
input_data = data['inputs']
output_data = data['outputs']

inputs = Dict(zip(input_names, [input_data]))
outputs = Dict(zip(output_name, [output_data]))

np.savez(fn, inputs=inputs, outputs=outputs)  # overwrite
