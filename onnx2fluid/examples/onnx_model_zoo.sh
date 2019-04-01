#! /usr/bin/env sh

# setopt SH_WORD_SPLIT # if zsh

base_url="https://s3.amazonaws.com/download.onnx/models/opset_9/"
convert_flags="-e -o /tmp/export/"
validate_flags1="/tmp/export/model.py"
validate_flags2="/tmp/export/__model__"

# alias http_get="wget -c" # if no aria2
alias http_get="aria2c -c -s8 -x8"
# alias python="python3" # if ...

bvlc_alexnet()
{
    bn_tar="bvlc_alexnet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for npz in "$bn_tar"/*.npz
    do
        echo "converting $npz ..."
        python convert_data_npz_0.py "$npz" data_0 prob_1 -s
        python -m onnx2fluid.validation $validate_flags1 -t "$npz"
        python -m onnx2fluid.validation $validate_flags2 -t "$npz"
    done
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir ..."
        python convert_data_pb_0.py "$pb_dir" data_0 prob_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

bvlc_googlenet()
{
    bn_tar="bvlc_googlenet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" data_0 prob_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

bvlc_reference_caffenet()
{
    bn_tar="bvlc_reference_caffenet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" data_0 prob_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

bvlc_reference_rcnn_ilsvrc13()
{
    bn_tar="bvlc_reference_rcnn_ilsvrc13"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" data_0 fc-rcnn_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz -p 0
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz -p 0
    done
}

inception_v1()
{
    bn_tar="inception_v1"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for npz in "$bn_tar"/*.npz
    do
        echo "converting $npz ..."
        python convert_data_npz_0.py "$npz" data_0 prob_1 -s
        python -m onnx2fluid.validation $validate_flags1 -t "$npz"
        python -m onnx2fluid.validation $validate_flags2 -t "$npz"
    done
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir ..."
        python convert_data_pb_0.py "$pb_dir" data_0 prob_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

inception_v2()
{
    bn_tar="inception_v2"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for npz in "$bn_tar"/*.npz
    do
        echo "converting $npz ..."
        python convert_data_npz_0.py "$npz" data_0 prob_1 -s
        python -m onnx2fluid.validation $validate_flags1 -t "$npz"
        python -m onnx2fluid.validation $validate_flags2 -t "$npz"
    done
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir ..."
        python convert_data_pb_0.py "$pb_dir" data_0 prob_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

resnet50()
{
    bn_tar="resnet50"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for npz in "$bn_tar"/*.npz
    do
        echo "converting $npz ..."
        python convert_data_npz_0.py "$npz" gpu_0/data_0 gpu_0/softmaxout_1 -s
        python -m onnx2fluid.validation $validate_flags1 -t "$npz"
        python -m onnx2fluid.validation $validate_flags2 -t "$npz"
    done
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir ..."
        python convert_data_pb_0.py "$pb_dir" gpu_0/data_0 gpu_0/softmaxout_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

shufflenet()
{
    bn_tar="shufflenet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir ..."
        python convert_data_pb_0.py "$pb_dir" gpu_0/data_0 gpu_0/softmaxout_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

squeezenet()
{
    bn_tar="squeezenet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" data_0 softmaxout_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

tiny_yolov2()
{
    bn_tar="tiny_yolov2"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model" -xy
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" image grid
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

vgg19()
{
    bn_tar="vgg19"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" data_0 prob_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}

zfnet512()
{
    bn_tar="zfnet512"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    http_get "$base_url$fn_tar"
    rm -rf "$bn_tar/"
    echo "extracting ..."
    tar xf "$fn_tar"

    python -m onnx2fluid $convert_flags "$fn_model"
    for pb_dir in "$bn_tar"/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" gpu_0/data_0 gpu_0/softmax_1
        python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
        python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
    done
}


bvlc_alexnet
bvlc_googlenet
bvlc_reference_caffenet
bvlc_reference_rcnn_ilsvrc13
inception_v1
inception_v2
resnet50
shufflenet
squeezenet # softmax bug
# tiny_yolov2 # not supported
vgg19
zfnet512
