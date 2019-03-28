#! /usr/bin/env sh

get_url="aria2c -c -s8 -x8"
base_url="https://s3.amazonaws.com/download.onnx/models/opset_9/"
flags="-e -o /tmp/export/"

bvlc_alexnet()
{
    bn_tar="bvlc_alexnet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for npz in $bn_tar/*.npz
    do
        echo "converting $npz ..."
        python convert_data_npz_0.py "$npz" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $npz
    done
    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir ..."
        python convert_data_pb_0.py "$pb_dir" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

bvlc_googlenet()
{
    bn_tar="bvlc_googlenet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

bvlc_reference_caffenet()
{
    bn_tar="bvlc_reference_caffenet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

bvlc_reference_rcnn_ilsvrc13()
{
    bn_tar="bvlc_reference_rcnn_ilsvrc13"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "data_0" "fc_rcnn_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

inception_v1()
{
    bn_tar="inception_v1"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for npz in $bn_tar/*.npz
    do
        echo "converting $npz ..."
        python convert_data_npz_0.py "$npz" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $npz
    done

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

inception_v2()
{
    bn_tar="inception_v2"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for npz in $bn_tar/*.npz
    do
        echo "converting $npz ..."
        python convert_data_npz_0.py "$npz" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $npz
    done

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

resnet50()
{
    bn_tar="resnet50"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for npz in $bn_tar/*.npz
    do
        echo "converting $npz ..."
        python convert_data_npz_0.py "$npz" "gpu_0/data_0" "gpu_0/softmaxout_1"
        python -m onnx2fluid $flags "$fn_model" -t $npz
    done

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "gpu_0/data_0" "gpu_0/softmaxout_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

shufflenet()
{
    bn_tar="shufflenet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "gpu_0/data_0" "gpu_0/softmaxout_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

squeezenet()
{
    bn_tar="squeezenet"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "data_0" "softmaxout_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

tiny_yolov2()
{
    bn_tar="tiny_yolov2"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "image" "grid"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz -x
    done
}

vgg19()
{
    bn_tar="vgg19"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "data_0" "prob_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
    done
}

zfnet512()
{
    bn_tar="zfnet512"
    fn_tar="$bn_tar.tar.gz"
    fn_model="$bn_tar/model.onnx"

    $get_url "$base_url$fn_tar"
    echo "extracting ..."
    tar xf "$fn_tar"

    for pb_dir in $bn_tar/*/
    do
        echo "converting $pb_dir"
        python convert_data_pb_0.py "$pb_dir" "gpu_0/data_0" "gpu_0/softmax_1"
        python -m onnx2fluid $flags "$fn_model" -t $(dirname "$pb_dir/x").npz
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
squeezenet
tiny_yolov2 # not supported
vgg19
zfnet512
