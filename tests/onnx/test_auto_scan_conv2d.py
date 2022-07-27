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
import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np
import unittest


class TestConv2dConvert(OPConvertAutoScanTest):
    """
    ONNX op: Conv
    OPset version: 7~15
    """

    def add_ignore_test_case(self, configs):
        config, attrs = configs
        # Warning: SAME_UPPER and SAME_LOWER does not yet support dynamic shapes
        if "SAME" in attrs["auto_pad"] and -1 in config["inputs_shape"][0]:
            return True
        else:
            return False

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=30), min_size=4, max_size=4))

        kernel_size = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=7), min_size=4, max_size=4))

        data_format = "NCHW"

        groups = draw(st.integers(min_value=1, max_value=4))
        muti1 = draw(st.integers(min_value=1, max_value=4))
        kernel_size[0] = groups * muti1
        input_shape[1] = kernel_size[1] * groups

        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=1, max_size=2))
        if len(strides) == 1:
            strides = strides[0]
            if strides > kernel_size[2]:
                strides = kernel_size[2]
            if strides > kernel_size[3]:
                strides = kernel_size[3]
            strides = [strides, strides]
        else:
            if strides[0] > kernel_size[2]:
                strides[0] = kernel_size[2]
            if strides[1] > kernel_size[3]:
                strides[1] = kernel_size[3]

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

        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=3), min_size=2, max_size=2))

        config = {
            "op_names": ["Conv"],
            "test_data_shapes": [input_shape, kernel_size],
            "test_data_types": [['float32'], ['float32']],
            "inputs_shape": [[-1, input_shape[1], -1, -1], kernel_size],
            "min_opset_version": 7,
            "inputs_name": ["x", "W"],
            "outputs_name": ["y"],
            "delta": 1e-4,
            "rtol": 1e-4
        }

        attrs = {
            "auto_pad": auto_pad,
            "dilations": dilations,
            "group": groups,
            "kernel_shape": kernel_size[2:],
            "pads": padding,
            "strides": strides,
        }

        # if autopad equal SAME_UPPER and SAME_LOWER, dilations only support 1
        if "SAME" in auto_pad:
            attrs["dilations"] = [1, 1]

        return (config, attrs)

    def test(self):
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
