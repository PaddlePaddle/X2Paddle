from .register import register
from x2paddle.core.util import *


def detectionoutput_shape(input_shape):
    return [[-1, 6]]


def detectionoutput_layer(inputs,
                          nms_param=None,
                          background_label_id=0,
                          share_location=True,
                          keep_top_k=100,
                          confidence_threshold=0.1,
                          input_shape=None,
                          name=None):
    nms_param_str = nms_param
    nms_param = {}
    part = nms_param_str.split(',')
    for s in part:
        if s == '':
            break
        else:
            name, obj = s.split(': ')
            if name == 'top_k':
                nms_param[name] = int(obj)
            else:
                nms_param[name] = float(obj)
    if nms_param is None:
        nms_param = {"nms_threshold": 0.3, "top_k": 10, "eta": 1.0}
    mbox_conf_flatten = inputs[1]
    mbox_priorbox = inputs[2]
    mbox_priorbox_list = fluid.layers.split(mbox_priorbox, 2, dim=1)
    pb = mbox_priorbox_list[0]
    pbv = mbox_priorbox_list[1]
    pb = fluid.layers.reshape(x=pb, shape=[-1, 4])
    pbv = fluid.layers.reshape(x=pbv, shape=[-1, 4])
    mbox_loc = inputs[0]
    mbox_loc = fluid.layers.reshape(x=mbox_loc, shape=[-1, pb.shape[0], 4])
    mbox_conf_flatten = fluid.layers.reshape(x=mbox_conf_flatten,
                                             shape=[0, pb.shape[0], -1])

    default = {"nms_threshold": 0.3, "top_k": 10, "eta": 1.0}
    fields = ['eta', 'top_k', 'nms_threshold']
    for f in default.keys():
        if f not in nms_param:
            nms_param[f] = default[f]
    out = fluid.layers.detection_output(
        scores=mbox_conf_flatten,
        loc=mbox_loc,
        prior_box=pb,
        prior_box_var=pbv,
        background_label=background_label_id,
        nms_threshold=nms_param["nms_threshold"],
        nms_top_k=nms_param["top_k"],
        keep_top_k=keep_top_k,
        score_threshold=confidence_threshold,
        nms_eta=nms_param["eta"])
    return out


def detectionoutput_weights(name, data=None):
    weights_name = []
    return weights_name


register(kind='DetectionOutput',
         shape=detectionoutput_shape,
         layer=detectionoutput_layer,
         weights=detectionoutput_weights)
