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

min_opset_version_map = {
    "Cosh": 9,
    "Cos": 7,
    "Atan": 7,
    "Asinh": 9,
    "Asin": 7,
    "Acosh": 9,
    "Acos": 7,
    "Exp": 7,
    "Floor": 7,
    "Tan": 7,
    "Ceil": 7,
    "Erf": 9,
    "Sin": 7,
    "Sinh": 9,
    "Tanh": 7,
    "Atanh": 9,
    "Sign": 9,
    "Softplus": 7,
    "Sigmoid": 7,
    "Neg": 7,
    "Softsign": 7,
}


class TestUnaryopsConcert(OPConvertAutoScanTest):
    """
    ONNX op: unary ops
    OPset version: 7~15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=30), min_size=3, max_size=5))

        input_dtype = draw(st.sampled_from(["float32"]))

        config = {
            "op_names": [
                "Cos", "Atan", "Asinh", "Asin", "Acosh", "Acos", "Cosh", "Exp",
                "Floor", "Tan", "Erf", "Sin", "Sinh", "Tanh", "Atanh", "Sign",
                "Softplus", "Sigmoid", "Neg", "Softsign"
            ],
            "test_data_shapes": [input_shape],
            "test_data_types": [input_dtype],
            "inputs_shape": [input_shape],
            "min_opset_version": 7,
            "inputs_name": ["x"],
            "outputs_name": ["y"],
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
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
