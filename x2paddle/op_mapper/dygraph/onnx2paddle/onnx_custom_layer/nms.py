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
from paddle.fluid import core

class NMS(object):
    def __init__(self, score_threshold, nms_top_k, nms_threshold):
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.nms_threshold = nms_threshold
        
    def __call__(self, bboxes, scores):
        attrs = ('background_label', -1, 
                 'score_threshold', self.score_threshold, 
                 'nms_top_k', self.nms_top_k, 
                 'nms_threshold', self.nms_threshold, 
                 'keep_top_k',  -1,
                 'nms_eta', 1.0,
                 'normalized', False)
        output, index, nms_rois_num = core.ops.multiclass_nms3(bboxes, scores,
                                                               None, *attrs)
        clas = paddle.slice(output, axes=[1], starts=[0], ends=[1])
        clas = paddle.cast(clas, dtype="int32")
        if bboxes.shape[0] == 1:
            batch = paddle.zeros_like(clas, dtype="int32")
        else:
            bboxes_count = bboxes.shape[1]
            bboxes_count_tensor = paddle.full_like(index, fill_value=bboxes_count, dtype="int32")
            batch = paddle.divide(index, bboxes_count_tensor)
            index = paddle.mod(index, bboxes_count_tensor)
        res = paddle.concat([batch, clas, index], axis=1)
        return res