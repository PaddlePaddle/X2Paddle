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

    def forward(self, inputs):
        """
        forward
        """
        x = torch.nn.functional.avg_pool1d(
            inputs,
            kernel_size=self.config["kernel_size"],
            stride=self.config["stride"],
            padding=self.config["padding"],
            ceil_mode=self.config["ceil_mode"],
            count_include_pad=self.config["count_include_pad"])
        return x


class TestAvgPool1dConvert(OPConvertAutoScanTest):
    """
    Torch API: torch.nn.functional.avg_pool1d
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=16, max_value=32),
                     min_size=3,
                     max_size=3))

        kernel_size = draw(st.integers(min_value=1, max_value=5))
        stride = draw(st.integers(min_value=1, max_value=5))
        padding = draw(st.integers(min_value=0, max_value=kernel_size // 2))
        ceil_mode = draw(st.booleans())
        count_include_pad = draw(st.booleans())

        config = {
            "op_names": ["avg_pool1d"],
            "test_data_shapes": [input_shape],
            "test_data_types": [['float32']],
            "inputs_shape": [[-1, input_shape[1], -1]],
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
            "count_include_pad": count_include_pad,
            "delta": 1e-4,
            "rtol": 1e-4,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
