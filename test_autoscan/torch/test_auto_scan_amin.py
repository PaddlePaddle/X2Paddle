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
        x = torch.amin(inputs,
                       dim=self.config["dim"],
                       keepdim=self.config["keepdim"])
        return x


class TestAmaxConvert(OPConvertAutoScanTest):
    """
    Torch API: torch.amin
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=1, max_value=64),
                     min_size=1,
                     max_size=5))

        dim_type = draw(st.sampled_from(["tuple", "int", None]))

        dim = None
        if dim_type == "int":
            dim = draw(
                st.integers(min_value=-len(input_shape),
                            max_value=len(input_shape) - 1))
        elif dim_type == "tuple":
            dim = draw(
                st.lists(st.integers(min_value=-len(input_shape),
                                     max_value=len(input_shape) - 1),
                         min_size=1,
                         max_size=len(input_shape),
                         unique_by=lambda x: x + len(input_shape)
                         if x < 0 else x))

        keepdim = draw(st.booleans())

        config = {
            "op_names": ["amin"],
            "test_data_shapes": [input_shape],
            "test_data_types": [['float32']],
            "inputs_shape": [input_shape],
            "dim": dim,
            "keepdim": keepdim
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
