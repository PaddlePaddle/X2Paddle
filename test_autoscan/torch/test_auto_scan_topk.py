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
        x = torch.topk(inputs,
                       k=self.config["k"],
                       dim=self.config["dim"],
                       largest=self.config["largest"],
                       sorted=self.config["sorted"])
        return x


class TestTopkConvert(OPConvertAutoScanTest):
    """
    Torch API: torch.topk
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=1, max_value=32),
                     min_size=1,
                     max_size=5))

        dim = draw(
            st.integers(min_value=-len(input_shape),
                        max_value=len(input_shape) - 1))

        k = draw(st.integers(min_value=1, max_value=input_shape[dim]))

        largest = draw(st.booleans())

        # Paddle will sort the result when using GPU.
        sorted_value = True
        if not torch.cuda.is_available():
            sorted_value = draw(st.booleans())

        config = {
            "op_names": ["topk"],
            "test_data_shapes": [input_shape],
            "test_data_types": [['float32']],
            "inputs_shape": [input_shape],
            "k": k,
            "dim": dim,
            "largest": largest,
            "sorted": sorted_value
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
