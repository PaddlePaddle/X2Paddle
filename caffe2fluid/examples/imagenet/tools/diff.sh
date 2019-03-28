#!/bin/bash

#function:
#   a tool used to:
#       1, convert a caffe model
#       2, do inference(only in fluid) using this model
#
#usage:
#   cd caffe2fluid/examples/imagenet && bash run.sh alexnet ./models/alexnet.prototxt ./models/alexnet.caffemodel ./models/alexnet.py ./models/alexnet.npy
#

#set -x


if [[ $# -lt 5 ]];then
    echo "usage:"
    echo "  bash $0 [model_name] [cf_prototxt_path] [cf_model_path] [pd_py_path] [pd_npy_path] [imagfile] [only_convert]"
    echo "  eg: bash $0 alexnet ./models/alexnet.prototxt ./models/alexnet.caffemodel ./models/alexnet.py ./models/alexnet.npy"
    exit 1
else
    model_name=$1
    cf_prototxt_path=$2
    cf_model_path=$3
    pd_py_path=$4
    pd_npy_path=$5
    only_convert=$7
fi
proto_file=$cf_prototxt_path
caffemodel_file=$cf_model_path
weight_file=$pd_npy_path
net_file=$pd_py_path

if [[ ! -e $proto_file ]];then
    echo "not found prototxt[$proto_file]"
    exit 1
fi

if [[ ! -e $caffemodel_file ]];then
    echo "not found caffemodel[$caffemodel_file]"
    exit 1
fi

if [[ ! -e $pd_model_path ]];then
    mkdir $pd_model_path
fi

PYTHON=`which python`
if [[ -z $PYTHON ]];then
    PYTHON=`which python`
fi
$PYTHON ../../convert.py \
        $proto_file \
        --caffemodel $caffemodel_file \
        --data-output-path $weight_file\
        --code-output-path $net_file

ret=$?
if [[ $ret -ne 0 ]];then
    echo "failed to convert caffe model[$cf_model_path]"
    exit $ret
else
    echo "succeed to convert caffe model[$cf_model_path] to fluid model[$pd_model_path]"
fi

if [[ -z $only_convert ]];then
    PYTHON=`which python`
    if [[ -z $PYTHON ]];then
        PYTHON=`which python`
    fi
    if [[ -n $6 ]];then
        imgfile=$6
    else
        imgfile="data/65.jpeg"
    fi
    #FIX ME:
    #   only look the first line in prototxt file for the name of this network, maybe not correct
    net_name=`grep "name" $proto_file | head -n1 | perl -ne 'if(/^name\s*:\s*\"([^\"]+)\"/){ print $1."\n";}'`
    if [[ -z $net_name ]];then
        net_name="MyNet"
    fi
    cmd="$PYTHON ./infer.py dump $net_file $weight_file $imgfile $net_name"
    echo $cmd
    eval $cmd
    ret=$?
fi
exit $ret
