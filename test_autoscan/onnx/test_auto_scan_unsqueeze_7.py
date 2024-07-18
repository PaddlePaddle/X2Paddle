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
import hypothesis.strategies as st
import numpy as np
import unittest


class TestUnsqueezeConvert(OPConvertAutoScanTest):
    """
    ONNX op: Unsqueeze
    OPset version: 7~12
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=6), min_size=2, max_size=5))

        input_dtype = draw(
            st.sampled_from(["int32", "int64", "float32", "float64"]))

        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))

        if len(input_shape) == 5:
            axis = [0]
        if len(input_shape) == 4:
            axis = [0, 1]
        if len(input_shape) == 3:
            axis = [1, 2, 3]
        if len(input_shape) == 2:
            if draw(st.booleans()):
                axis = [0, 1, 2]
            else:
                axis = [1, 3]

        config = {
            "op_names": ["Unsqueeze"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[input_dtype]],
            "inputs_shape": [input_shape],
            "min_opset_version": 7,
            "max_opset_version": 12,
            "inputs_name": ["x"],
            "outputs_name": ["y"],
            "delta": 1e-4,
            "rtol": 1e-4
        }

        attrs = {"axes": axis, }

        return (config, attrs)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
