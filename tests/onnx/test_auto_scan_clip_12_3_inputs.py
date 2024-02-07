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
import random
from onnxbase import randtool


class TestClipConvert(OPConvertAutoScanTest):
    """
    ONNX op: Clip
    OPset version: 12~15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        input_dtype = draw(st.sampled_from(["float32", "int64"]))

        def generator_min():
            input_data = randtool("float", -10, -1, [1])
            return input_data

        def generator_max():
            input_data = randtool("float", 0, 10, [1])
            return input_data

        config = {
            "op_names": ["Clip"],
            "test_data_shapes": [input_shape, generator_min, generator_max],
            "test_data_types": [[input_dtype], [input_dtype], [input_dtype]],
            "inputs_shape": [],
            "min_opset_version": 12,
            "max_opset_version": 15,
            "inputs_name": ["x", "min", "max"],
            "outputs_name": ["y"],
            "delta": 1e-4,
            "rtol": 1e-4,
        }

        attrs = {}

        return (config, attrs)

    def test(self):
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
