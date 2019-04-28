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
		python convert_data_npz.py "$npz" data_0 prob_1 -s
		python -m onnx2fluid.validation $validate_flags1 -t "$npz"
		python -m onnx2fluid.validation $validate_flags2 -t "$npz"
	done
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data_0 prob_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_pb.py "$pb_dir" data_0 prob_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_pb.py "$pb_dir" data_0 prob_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_pb.py "$pb_dir" data_0 fc-rcnn_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

densenet121()
{
	bn_tar="densenet121"
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
		python convert_data_npz.py "$npz" data_0 fc6_1 -s
		python -m onnx2fluid.validation $validate_flags1 -t "$npz"
		python -m onnx2fluid.validation $validate_flags2 -t "$npz"
	done
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir"
		python convert_data_pb.py "$pb_dir" data_0 fc6_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

   rm -rf "$bn_tar/"
}

emotion_ferplus()
{
	bn_tar="emotion_ferplus"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/model.onnx"

	http_get "https://onnxzoo.blob.core.windows.net/models/opset_8/emotion_ferplus/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid $convert_flags "$fn_model" -y
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" Input3 Plus692_Output_0
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_npz.py "$npz" data_0 prob_1 -s
		python -m onnx2fluid.validation $validate_flags1 -t "$npz"
		python -m onnx2fluid.validation $validate_flags2 -t "$npz"
	done
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data_0 prob_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_npz.py "$npz" data_0 prob_1 -s
		python -m onnx2fluid.validation $validate_flags1 -t "$npz"
		python -m onnx2fluid.validation $validate_flags2 -t "$npz"
	done
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data_0 prob_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

mobilenet()
{
	bn_tar="mobilenetv2-1.0"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/$bn_tar.onnx"

	http_get "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid $convert_flags "$fn_model" -y
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data mobilenetv20_output_flatten0_reshape0
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

resnet18()
{
	bn_tar="resnet18v1"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/$bn_tar.onnx"

	http_get "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid $convert_flags "$fn_model" -y
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data resnetv15_dense0_fwd
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_npz.py "$npz" gpu_0/data_0 gpu_0/softmaxout_1 -s
		python -m onnx2fluid.validation $validate_flags1 -t "$npz"
		python -m onnx2fluid.validation $validate_flags2 -t "$npz"
	done
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" gpu_0/data_0 gpu_0/softmaxout_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

resnet100_arcface()
{
	bn_tar="resnet100"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/$bn_tar.onnx"

	http_get "https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid -o /tmp/export/ "$fn_model" -y
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data fc1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

resnet101_duc()
{
	bn_tar="ResNet101_DUC_HDC"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/$bn_tar.onnx"

	http_get "https://s3.amazonaws.com/onnx-model-zoo/duc/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid $convert_flags "$fn_model" -y
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data seg_loss
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

resnet152()
{
	bn_tar="resnet152v2"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/$bn_tar.onnx"

	http_get "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid $convert_flags "$fn_model" -y
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data resnetv27_dense0_fwd
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_pb.py "$pb_dir" gpu_0/data_0 gpu_0/softmax_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_pb.py "$pb_dir" data_0 softmaxout_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

squeezenet1v1()
{
	bn_tar="squeezenet1.1"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/$bn_tar.onnx"

	http_get "https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid $convert_flags "$fn_model"
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data squeezenet0_flatten0_reshape0
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

ssd()
{
	bn_tar="ssd"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/model.onnx"

	http_get "https://onnxzoo.blob.core.windows.net/models/opset_10/ssd/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	mkdir "$bn_tar"
	tar xf "$fn_tar" -C "$bn_tar"/

	python -m onnx2fluid $convert_flags "$fn_model"
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" image bboxes,labels,scores
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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

	python -m onnx2fluid $convert_flags "$fn_model" -y
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir"
		python convert_data_pb.py "$pb_dir" image grid
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

vgg16bn()
{
	bn_tar="vgg16-bn"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/$bn_tar.onnx"

	http_get "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid $convert_flags "$fn_model" -y
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" data vgg0_dense2_fwd
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_pb.py "$pb_dir" data_0 prob_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}

yolov3()
{
	bn_tar="yolov3"
	fn_tar="$bn_tar.tar.gz"
	fn_model="$bn_tar/model.onnx"

	http_get "https://onnxzoo.blob.core.windows.net/models/opset_10/yolov3/$fn_tar"
	rm -rf "$bn_tar/"
	echo "extracting ..."
	tar xf "$fn_tar"

	python -m onnx2fluid $convert_flags "$fn_model" -x #
	for pb_dir in "$bn_tar"/*/
	do
		echo "converting $pb_dir ..."
		python convert_data_pb.py "$pb_dir" input_1:01,image_shape:01 yolonms_layer_1/ExpandDims_1:0,yolonms_layer_1/ExpandDims_3:0,yolonms_layer_1/concat_2:0
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
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
		python convert_data_pb.py "$pb_dir" gpu_0/data_0 gpu_0/softmax_1
		python -m onnx2fluid.validation $validate_flags1 -t $(dirname "$pb_dir/x").npz
		python -m onnx2fluid.validation $validate_flags2 -t $(dirname "$pb_dir/x").npz
	done

	rm -rf "$bn_tar/"
}


bvlc_alexnet
bvlc_googlenet
bvlc_reference_caffenet
bvlc_reference_rcnn_ilsvrc13
densenet121
emotion_ferplus # not supported
inception_v1
inception_v2
mobilenet
resnet18
resnet50
resnet100_arcface
resnet101_duc
resnet152
shufflenet
squeezenet # softmax bug
squeezenet1v1
ssd # version not supported
tiny_yolov2 # not supported
vgg16bn
vgg19
yolov3 # malformed model ?
zfnet512
