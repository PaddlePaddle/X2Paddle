目前X2Paddle支持40+的TensorFlow OP，40+的Caffe Layer，覆盖了大部分CV分类模型常用的操作。我们在如下模型列表中测试了X2Paddle的转换

# TensorFlow

| 模型 | 代码 |
|------|----------|
| SqueezeNet | [code](https://github.com/tensorflow/tpu/blob/master/models/official/squeezenet/squeezenet_model.py)|
| MobileNet_V1 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) |
| MobileNet_V2 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) |
| ShuffleNet | [code](https://github.com/TropComplique/shufflenet-v2-tensorflow) |
| mNASNet | [code](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) |
| EfficientNet | [code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| Inception_V4 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) |
| Inception_ResNet_V2 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) |
| VGG16 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) |
| ResNet_V1_101 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) |
| ResNet_V2_101 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) |

# Caffe

| 模型 | 代码 |
|-------|--------|
| SqueezeNet | [code](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) |
| MobileNet_V1 | [code](https://github.com/shicai/MobileNet-Caffe) |
| MobileNet_V2 | [code](https://github.com/shicai/MobileNet-Caffe) |
| ShuffleNet | [code](https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe/releases/tag/v0.1.0) |
| mNASNet | [code](https://github.com/LiJianfei06/MnasNet-caffe) |
| MTCNN | [code](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv1/model) |

# ONNX

| 模型 | 来源 | operator version|
|-------|--------|---------|
| Resnet18 | [torchvison.model.resnet18](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9|
| Resnet34 | [torchvison.model.resnet34](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9|
| Resnet50 | [torchvison.model.resnet50](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9|
| Resnet101 | [torchvison.model.resnet101](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9|
| Vgg11 | [torchvison.model.vgg11](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9|
| Vgg11_bn | [torchvison.model.vgg11_bn](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9|
| Vgg19| [torchvison.model.vgg19](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9|
| Densenet121 | [torchvison.model.densenet121](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py) |9|
| Alexnet | [torchvison.model.alexnet](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py) |9|
| Shufflenet | [onnx official](https://github.com/onnx/models/tree/master/vision/classification/shufflenet) |9|
| Inception_v2 | [onnx official](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2) |9|
| Mobilenet_v2 | [pytorch(personal practice)](https://github.com/tonylins/pytorch-mobilenet-v2) |9|

目前onnx2paddle主要支持onnx operator version 9；
如何将torchvison或者个人开发者写的pytroch model转换成onnx model:
```
import torch
import torchvision

#根据不同模型调整输入的shape
dummy_input = torch.randn(1, 3, 224, 224)

#预训练后的pytorch model
resnet18 = torchvision.models.resnet18(pretrained=True)

#"resnet18.onnx"为onnx model的存储路径，1.1
torch.onnx.export(resnet18, dummy_input, "resnet18.onnx",verbose=True)

```
