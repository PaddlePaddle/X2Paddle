# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid


class DetectionOutput(object):
    def __init__(self, nms_threshold, nms_top_k, keep_top_k, nms_eta,
                 score_threshold, background_label):
        self.detection_output_layer_attrs = {
            "background_label": background_label,
            "nms_threshold": nms_threshold,
            "nms_top_k": nms_top_k,
            "keep_top_k": keep_top_k,
            "score_threshold": score_threshold,
            "nms_eta": nms_eta
        }

    def __call__(self, x0, x1, x2):
        priorbox_list = paddle.split(x2, num_or_sections=2, axis=1)
        pb = priorbox_list[0]
        pbv = priorbox_list[1]
        pb = paddle.reshape(x=pb, shape=[-1, 4])
        pbv = paddle.reshape(x=pbv, shape=[-1, 4])
        pb_dim = fluid.layers.shape(pb)[0]
        loc = paddle.reshape(x0, shape=[-1, pb_dim, 4])
        conf_flatten = paddle.reshape(x1, shape=[0, pb_dim, -1])
        out = fluid.layers.detection_output(
            loc=loc,
            scores=conf_flatten,
            prior_box=pb,
            prior_box_var=pbv,
            **self.detection_output_layer_attrs)
        return out
