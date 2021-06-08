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
from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper


def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   nms_top_k,
                   keep_top_k,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=-1,
                   return_index=False,
                   return_rois_num=True,
                   rois_num=None,
                   name=None):
    helper = LayerHelper('multiclass_nms3', **locals())

    if in_dygraph_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
                 nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
                 'normalized', normalized)
        output, index, nms_rois_num = core.ops.multiclass_nms3(bboxes, scores,
                                                               rois_num, *attrs)
        if not return_index:
            index = None
        return output, nms_rois_num, index

    else:
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int')

        inputs = {'BBoxes': bboxes, 'Scores': scores}
        outputs = {'Out': output, 'Index': index}

        if rois_num is not None:
            inputs['RoisNum'] = rois_num

        if return_rois_num:
            nms_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            outputs['NmsRoisNum'] = nms_rois_num

        helper.append_op(
            type="multiclass_nms3",
            inputs=inputs,
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'nms_top_k': nms_top_k,
                'nms_threshold': nms_threshold,
                'keep_top_k': keep_top_k,
                'nms_eta': nms_eta,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True
        index.stop_gradient = True
        if not return_index:
            index = None
        if not return_rois_num:
            nms_rois_num = None

        return output, nms_rois_num, index


class NMS(object):
    def __init__(self, score_threshold, keep_top_k, nms_threshold):
        self.score_threshold = score_threshold
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold

    def __call__(self, bboxes, scores):
        attrs = {
            'background_label': -1,
            'score_threshold': self.score_threshold,
            'nms_top_k': -1,
            'nms_threshold': self.nms_threshold,
            'keep_top_k': self.keep_top_k,
            'nms_eta': 1.0,
            'normalized': False,
            'return_index': True
        }
        output, nms_rois_num, index = multiclass_nms(bboxes, scores, **attrs)
        clas = paddle.slice(output, axes=[1], starts=[0], ends=[1])
        clas = paddle.cast(clas, dtype="int64")
        index = paddle.cast(index, dtype="int64")
        if bboxes.shape[0] == 1:
            batch = paddle.zeros_like(clas, dtype="int64")
        else:
            bboxes_count = bboxes.shape[1]
            batch = paddle.divide(index, bboxes_count)
            index = paddle.mod(index, bboxes_count)
        res = paddle.concat([batch, clas, index], axis=1)
        return res
