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


class TestAveragePoolConvert(OPConvertAutoScanTest):
    """
    ONNX op: AveragePool
    OPset version: 10~15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=4, max_size=4))

        kernel_size = draw(
            st.lists(
                st.integers(
                    min_value=7, max_value=10), min_size=2, max_size=2))

        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=2, max_size=2))

        if draw(st.booleans()):
            auto_pad = "NOTSET"
            padding = None
            if draw(st.booleans()):
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=2,
                        max_size=2))
                padding = [0, 0] + padding
            else:
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=4,
                        max_size=4))
        else:
            auto_pad = draw(
                st.sampled_from(
                    ["SAME_LOWER", "SAME_UPPER", "VALID", "NOTSET"]))
            padding = None

        if draw(st.booleans()):
            ceil_mode = 0
        else:
            ceil_mode = 1
        if padding == "VALID":
            ceil_mode = False

        config = {
            "op_names": ["AveragePool"],
            "test_data_shapes": [input_shape],
            "test_data_types": [["float32"], ],
            "inputs_shape": [],
            "min_opset_version": 10,
            "max_opset_version": 15,
            "inputs_name": ["x"],
            "outputs_name": ["y"],
            "delta": 1e-4,
            "rtol": 1e-4
        }

        attrs = {
            "auto_pad": auto_pad,
            "ceil_mode": ceil_mode,
            "kernel_shape": kernel_size,
            "pads": padding,
            "strides": strides,
        }
        return (config, attrs)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
