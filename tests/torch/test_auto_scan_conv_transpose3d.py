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
        x = torch.nn.functional.conv_transpose3d(
            inputs,
            weight,
            stride=self.config["stride"],
            padding=self.config["padding"],
            output_padding=self.config["output_padding"],
            dilation=self.config["dilation"],
            groups=self.config["groups"])
        return x


class TestConvtranspose3dConvert(OPConvertAutoScanTest):
    """
    Torch API: torch.nn.functional.conv_transpose3d
    """

    def add_ignore_test_case(self, configs):
        config, models = configs
        result = False
        # Warning: "same" padding mode doesn’t support any stride values other than 1
        if isinstance(config["stride"], list):
            if config["padding"] == "same" and (config["stride"][0] > 1 or
                                                config["stride"][1] > 1):
                result = True
        else:
            if config["padding"] == "same" and config["stride"] > 1:
                result = True
        # Warning: output padding must be smaller than either stride or dilation
        if isinstance(config["stride"], list):
            stride = config["stride"][0]
        else:
            stride = config["stride"]
        if isinstance(config["dilation"], list):
            dilation = config["dilation"][0]
        else:
            dilation = config["dilation"]
        if isinstance(config["output_padding"], int):
            if config["output_padding"] >= stride or config[
                    "output_padding"] >= dilation:
                result = True
        else:
            if config["output_padding"][0] >= stride or config[
                    "output_padding"][1] >= stride or config["output_padding"][
                        2] >= stride or config["output_padding"][
                            0] >= dilation or config["output_padding"][
                                1] >= dilation or config["output_padding"][
                                    2] >= dilation:
                result = True
        return result

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=5, max_size=5))
        # BS = 1
        input_shape[0] = 1
        kernel_size = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=7), min_size=5, max_size=5))

        groups = draw(st.integers(min_value=1, max_value=4))
        muti1 = draw(st.integers(min_value=1, max_value=4))
        kernel_size[0] = groups * muti1
        input_shape[1] = kernel_size[0]

        strides_type = draw(st.sampled_from(["list", "int"]))

        strides = None
        if strides_type == "int":
            strides = draw(st.integers(min_value=1, max_value=5))
            if strides > kernel_size[2]:
                strides = kernel_size[2]
            if strides > kernel_size[3]:
                strides = kernel_size[3]
            if strides > kernel_size[4]:
                strides = kernel_size[4]
        else:
            strides = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=3,
                    max_size=3))
            if strides[0] > kernel_size[2]:
                strides[0] = kernel_size[2]
            if strides[1] > kernel_size[3]:
                strides[1] = kernel_size[3]
            if strides[2] > kernel_size[4]:
                strides[2] = kernel_size[4]

        padding_type = draw(st.sampled_from(["int", "tuple"]))
        padding = None
        if padding_type == "int":
            padding = draw(st.integers(min_value=1, max_value=3))
        else:
            padding = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=3,
                    max_size=3))

        output_padding_type = draw(st.sampled_from(["int", "tuple"]))
        output_padding = None
        if output_padding_type == "int":
            output_padding = draw(st.integers(min_value=0, max_value=0))
        else:
            output_padding = draw(
                st.lists(
                    st.integers(
                        min_value=0, max_value=0),
                    min_size=3,
                    max_size=3))

        dilations_type = draw(st.sampled_from(["int", "tuple"]))
        dilations = None
        if dilations_type == "int":
            dilations = draw(st.integers(min_value=1, max_value=3))
        else:
            dilations = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=3),
                    min_size=3,
                    max_size=3))

        config = {
            "op_names": ["conv_transpose3d"],
            "test_data_shapes": [input_shape, kernel_size],
            "test_data_types": [['float32'], ['float32']],
            "inputs_shape": [[-1, input_shape[1], -1, -1, -1], kernel_size],
            "dilation": dilations,
            "groups": groups,
            "padding": padding,
            "output_padding": output_padding,
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
