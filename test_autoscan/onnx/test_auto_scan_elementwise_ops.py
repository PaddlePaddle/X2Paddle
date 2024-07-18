# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from auto_scan_test import OPConvertAutoScanTest
from hypothesis import reproduce_failure
from onnxbase import randtool
import hypothesis.strategies as st
import numpy as np
import unittest

min_opset_version_map = {
    "Add": 7,
    "Sub": 7,
    "Div": 7,
    "Mul": 7,
    "Pow": 7,
    "Max": 9,
    "Min": 9,
}


class TestElementwiseopsConvert(OPConvertAutoScanTest):
    """
    ONNX op: elementwise ops
    OPset version: 7~15
    """

    def sample_convert_config(self, draw):
        input1_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=2, max_size=4))

        if draw(st.booleans()):
            input2_shape = [input1_shape[-1]]
        else:
            input2_shape = input1_shape

        def generator_data():
            input_data = randtool("float", -5.0, 5.0, input2_shape)
            input_data[abs(input_data) < 1.0] = 1.0
            return input_data

        input_dtype = draw(st.sampled_from(["float32", "float64"]))

        config = {
            "op_names": ["Add", "Sub", "Div", "Mul", "Pow", "Max", "Min"],
            "test_data_shapes": [input1_shape, generator_data],
            "test_data_types": [[input_dtype], [input_dtype]],
            "inputs_shape": [],
            "min_opset_version": 7,
            "inputs_name": ["x", "y"],
            "outputs_name": ["z"],
            "delta": 1e-4,
            "rtol": 1e-4
        }
        min_opset_versions = list()
        for op_name in config["op_names"]:
            min_opset_versions.append(min_opset_version_map[op_name])
        config["min_opset_version"] = min_opset_versions

        attrs = {}

        return (config, attrs)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
