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

from auto_scan_test import OPConvertAutoScanTest, BaseNet
from hypothesis import reproduce_failure
import hypothesis.strategies as st
import torch
import numpy as np
import unittest


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, weight):
        """
        forward
        """
        x = torch.nn.functional.conv1d(
            inputs,
            weight,
            stride=self.config["stride"],
            padding=self.config["padding"],
            dilation=self.config["dilation"],
            groups=self.config["groups"])
        return x


class TestConv1dConvert(OPConvertAutoScanTest):
    """
    Torch API: torch.nn.functional.conv1d
    """

    def add_ignore_test_case(self, configs):
        config, models = configs
        result = False
        # Warning: "same" padding mode doesn’t support any stride values other than 1
        if isinstance(config["stride"], list):
            if config["padding"] == "same" and config["stride"][0] > 1:
                result = True
        else:
            if config["padding"] == "same" and config["stride"] > 1:
                result = True
        return result

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=15, max_value=25), min_size=3, max_size=3))

        kernel_size = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=3, max_size=3))

        groups = draw(st.integers(min_value=1, max_value=4))
        muti1 = draw(st.integers(min_value=1, max_value=4))
        kernel_size[0] = groups * muti1
        input_shape[1] = kernel_size[1] * groups

        strides_type = draw(st.sampled_from(["list", "int"]))

        strides = None
        if strides_type == "int":
            strides = draw(st.integers(min_value=1, max_value=5))
            if strides > kernel_size[2]:
                strides = kernel_size[2]
        else:
            strides = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=1,
                    max_size=1))
            if strides[0] > kernel_size[2]:
                strides[0] = kernel_size[2]

        # calculate torch version
        version = torch.__version__
        v0, v1, v2 = version.split('.')
        # Avoid the situation where the version is equal to 1.7.0+cu101
        if '+' in v2:
            v2 = v2.split('+')[0]
        version_sum = int(v0) * 100 + int(v1) * 10 + int(v2)
        # Only torch >= 1.9.0, padding can be string dtype
        is_padding_str = False
        if version_sum >= 190:
            is_padding_str = True
        padding = None
        if draw(st.booleans()) and is_padding_str:
            padding = draw(st.sampled_from(["valid", "same"]))
        else:
            padding = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=1,
                    max_size=1))

        dilations_type = draw(st.sampled_from(["int", "tuple"]))
        dilations = None
        if dilations_type == "int":
            dilations = draw(st.integers(min_value=1, max_value=3))
        else:
            dilations = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=3),
                    min_size=1,
                    max_size=1))
        if padding == "same":
            dilations = 1

        config = {
            "op_names": ["conv1d"],
            "test_data_shapes": [input_shape, kernel_size],
            "test_data_types": [['float32'], ['float32']],
            "inputs_shape": [[-1, input_shape[1], -1], kernel_size],
            "dilation": dilations,
            "groups": groups,
            "padding": padding,
            "stride": strides,
            "delta": 1e-4,
            "rtol": 1e-4,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
