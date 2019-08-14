#!/bin/bash

if [[ $# -lt 1 ]];then
    echo "Error usage! The first parameter is the frame to be compared, e.g. caffe, tensorflow, onnx"
fi

if [[ $1 == 'caffe' ]];then
    if [[ $# -ne 6 ]];then
        echo "usage:"
        echo "  bash $0 [frame_type] [fluid_param_path] [imagfile] [save_path] [cf_prototxt_path] [cf_model_path]"
        echo "  eg: bash $0 caffe ./models/alexnet/inference_model ./images/test.jpg ./models ./models/alexnet.prototxt ./models/alexnet.caffemodel"
        exit 1
    fi
    PYTHON=`which python`
    $PYTHON ./infer.py caffe \
            $5 \
            $6 \
            $3 \
            $4
    $PYTHON ./infer.py paddle \
            $2 \
            $3 \
            $4
    caffe_path=$4"/results.caffe"
    paddle_path=$4"/results.paddle"
    echo ${caffe_path}
    echo ${paddle_path}
    $PYTHON ./compare.py ${caffe_path} ${paddle_path}
elif [[ $1 == 'tensorflow' ]];then
    echo "need to add code"
elif [[ $1 == 'onnx' ]];then
    echo "need to add code"
else
    echo "The first parameter must define the frame type."
fi
