# X2Paddle模型测试库
> 目前X2Paddle支持50+的TensorFlow OP，40+的Caffe Layer，覆盖了大部分CV分类模型常用的操作。我们在如下模型列表中测试了X2Paddle的转换。

**注：** 受限于不同框架的差异，部分模型可能会存在目前无法转换的情况，如TensorFlow中包含控制流的模型，NLP模型等。对于CV常见的模型，如若您发现无法转换或转换失败，存在较大diff等问题，欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/X2Paddle/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会及时跟进：）

## TensorFlow

| 模型 | 代码 | 备注 |
|------|----------|------|
| SqueezeNet | [code](https://github.com/tensorflow/tpu/blob/master/models/official/squeezenet/squeezenet_model.py)|-|
| MobileNet_V1 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) |-|
| MobileNet_V2 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) |-|
| ShuffleNet | [code](https://github.com/TropComplique/shufflenet-v2-tensorflow) |-|
| mNASNet | [code](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) |-|
| EfficientNet | [code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |-|
| Inception_V3 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) |-|
| Inception_V4 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) |-|
| Inception_ResNet_V2 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) |-|
| VGG16 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) |-|
| ResNet_V1_101 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) |-|
| ResNet_V2_101 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) |-|
| UNet | [code1](https://github.com/jakeret/tf_unet )/[code2](https://github.com/lyatdawn/Unet-Tensorflow) |-|
|MTCNN | [code](https://github.com/AITTSMD/MTCNN-Tensorflow) |-|
|YOLO-V3| [code](https://github.com/YunYang1994/tensorflow-yolov3) | 转换需要关闭NHWC->NCHW的优化，见[文档Q2](FAQ.md) |
| FALSR | [code](https://github.com/xiaomi-automl/FALSR) | - |
| DCSCN | [code](https://modelzoo.co/model/dcscn-super-resolution) | - |

## Caffe

| 模型 | 代码 |
|-------|--------|
| SqueezeNet | [code](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) |
| MobileNet_V1 | [code](https://github.com/shicai/MobileNet-Caffe) |
| MobileNet_V2 | [code](https://github.com/shicai/MobileNet-Caffe) |
| ShuffleNet_v2 | [code](https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe/releases/tag/v0.1.0) |
| InceptionV3 | [code](https://github.com/soeaver/caffe-model/blob/master/cls/inception/) |
| InceptionV4 | [code](https://github.com/soeaver/caffe-model/blob/master/cls/inception/) |
| mNASNet | [code](https://github.com/LiJianfei06/MnasNet-caffe) |
| MTCNN | [code](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv1/model) |
| Mobilenet_SSD | [code](https://github.com/chuanqi305/MobileNet-SSD) |
| ResNet18 | [code](https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet/blob/master/deploy.prototxt) |
| ResNet50 | [code](https://github.com/soeaver/caffe-model/blob/master/cls/resnet/deploy_resnet50.prototxt) |
| Unet | [code](https://github.com/jolibrain/deepdetect/blob/master/templates/caffe/unet/deploy.prototxt) |
| VGGNet | [code](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt) |
| FaceDetection | [code](https://github.com/ShiqiYu/libfacedetection/blob/master/models/caffe/yufacedetectnet-open-v1.prototxt) |





## ONNX
**注：** 部分模型来源于PyTorch，PyTorch的转换可参考[pytorch_to_onnx.md](pytorch_to_onnx.md)

| 模型 | 来源 | operator version|备注|
|-------|--------|---------|---------|
| ResNet18 | [torchvison.model.resnet18](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9|
| ResNet34 | [torchvison.model.resnet34](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9|
| ResNet50 | [torchvison.model.resnet50](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9|
| ResNet101 | [torchvison.model.resnet101](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9|
| VGG11 | [torchvison.model.vgg11](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9|
| VGG11_bn | [torchvison.model.vgg11_bn](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9|
| VGG19| [torchvison.model.vgg19](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9|
| DenseNet121 | [torchvison.model.densenet121](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py) |9|
| AlexNet | [torchvison.model.alexnet](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py) |9|
| ShuffleNet | [onnx official](https://github.com/onnx/models/tree/master/vision/classification/shufflenet) |9|
| Inception_V2 | [onnx official](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2) |9|
| MobileNet_V2 | [pytorch(personal practice)](https://github.com/tonylins/pytorch-mobilenet-v2) |9|
| mNASNet | [pytorch(personal practice)](https://github.com/rwightman/gen-efficientnet-pytorch) |9|
| EfficientNet | [pytorch(personal practice)](https://github.com/rwightman/gen-efficientnet-pytorch) |9|
| SqueezeNet | [onnx official](https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz) |9|
|Ultra-Light-Fast-Generic-Face-Detector-1MB| [onnx_model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/models/onnx)|9 |
|BERT| [pytorch(huggingface)](https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb)|11|转换时需指定input shape，见[文档Q3](FAQ.md)|
|GPT2| [pytorch(huggingface)](https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb)|11|转换时需指定input shape，见[文档Q3](FAQ.md)|
