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

try:
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
            pb_dim = paddle.shape(pb)[0]
            loc = paddle.reshape(x0, shape=[-1, pb_dim, 4])
            conf_flatten = paddle.reshape(x1, shape=[0, pb_dim, -1])
            out = fluid.layers.detection_output(
                loc=loc,
                scores=conf_flatten,
                prior_box=pb,
                prior_box_var=pbv,
                **self.detection_output_layer_attrs)
            return out

except ImportError:

    from paddle import _C_ops
    from paddle.base.layer_helper import LayerHelper

    # ref: Paddle/test/legacy_test/test_multiclass_nms_op.py
    def multiclass_nms3(
        bboxes,
        scores,
        rois_num=None,
        score_threshold=0.3,
        nms_top_k=1000,
        keep_top_k=100,
        nms_threshold=0.3,
        normalized=True,
        nms_eta=1.0,
        background_label=-1,
        return_index=True,
        return_rois_num=True,
        name=None,
    ):
        helper = LayerHelper('multiclass_nms3', **locals())

        if paddle.in_dynamic_mode():
            attrs = (
                score_threshold,
                nms_top_k,
                keep_top_k,
                nms_threshold,
                normalized,
                nms_eta,
                background_label,
            )
            output, index, nms_rois_num = _C_ops.multiclass_nms3(
                bboxes, scores, rois_num, *attrs)
            if not return_index:
                index = None
            return output, index, nms_rois_num
        else:
            output = helper.create_variable_for_type_inference(
                dtype=bboxes.dtype)
            index = helper.create_variable_for_type_inference(dtype='int32')

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
                    'normalized': normalized,
                },
                outputs=outputs,
            )
            output.stop_gradient = True
            index.stop_gradient = True
            if not return_index:
                index = None
            if not return_rois_num:
                nms_rois_num = None

            return output, nms_rois_num, index

    def detection_output(loc,
                         scores,
                         prior_box,
                         prior_box_var,
                         background_label=0,
                         nms_threshold=0.3,
                         nms_top_k=400,
                         keep_top_k=200,
                         score_threshold=0.01,
                         nms_eta=1.0,
                         return_index=False):

        helper = paddle.base.layer_helper.LayerHelper("detection_output",
                                                      **locals())
        decoded_box = paddle.vision.ops.box_coder(
            prior_box=prior_box,
            prior_box_var=prior_box_var,
            target_box=loc,
            code_type='decode_center_size')
        scores = paddle.nn.functional.activation.softmax(x=scores)
        scores = paddle.transpose(scores, perm=[0, 2, 1])
        scores.stop_gradient = True

        output, _, _ = multiclass_nms3(
            bboxes=decoded_box,
            scores=scores,
            rois_num=None,
            score_threshold=score_threshold,
            nms_top_k=nms_top_k,
            keep_top_k=keep_top_k,
            nms_threshold=nms_threshold,
            normalized=True,
            nms_eta=nms_eta,
            background_label=background_label,
            return_index=return_index,
            return_rois_num=True,
            name=None,
        )
        return output

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
            pb_dim = paddle.shape(pb)[0]
            loc = paddle.reshape(x0, shape=[-1, pb_dim, 4])
            conf_flatten = paddle.reshape(x1, shape=[0, pb_dim, -1])
            out = detection_output(loc=loc,
                                   scores=conf_flatten,
                                   prior_box=pb,
                                   prior_box_var=pbv,
                                   **self.detection_output_layer_attrs)
            return out
