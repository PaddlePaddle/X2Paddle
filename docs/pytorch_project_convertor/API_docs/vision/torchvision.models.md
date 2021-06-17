## [torchvision.models](https://pytorch.org/vision/stable/models.html?highlight=torchvision%20models)
目前PaddlePaddle官方提供的模型参数与PyTorch不一致，为此X2Paddle提供了一套与torchvision模型参数一致且使用方式一致的模型库，以resnet18为例，具体使用方式如下:

```python
from x2paddle import models
# 构造权重随机初始化的模型：
resnet18 = models.resnet18_pth()
x = paddle.rand([1, 3, 224, 224])
out = model(x)

# 构造预训练模型：
resnet18 = models.resnet18_pth(pretrained=True)
x = paddle.rand([1, 3, 224, 224])
out = model(x)
```

目前支持的模型为：
| PyTorch模型                                                  | Paddle模型                       |
| ------------------------------------------------------------ | -------------------------------- |
| [torchvision.models.resnet18](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18) | [x2paddle.models.resnet18_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L310)     |
| [torchvision.models.resnet34](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet34) | [x2paddle.models.resnet34_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L322)     |
| [torchvision.models.resnet50](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet50) | [x2paddle.models.resnet50_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L334)     |
| [torchvision.models.resnet101](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet101) | [x2paddle.models.resnet101_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L346)    |
| [torchvision.models.resnet152](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet152) | [x2paddle.models.resnet152_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L358)    |
| [torchvision.models.resnext50_32x4d](https://pytorch.org/vision/stable/models.html#torchvision.models.resnext50_32x4d) | [x2paddle.models.resnext50_32x4d_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L370)  |
| [torchvision.models.resnext101_32x8d](https://pytorch.org/vision/stable/models.html#torchvision.models.resnext101_32x8d) | [x2paddle.resnext101_32x8d_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L385)        |
| [torchvision.models.wide_resnet50_2](https://pytorch.org/vision/stable/models.html#torchvision.models.wide_resnet50_2) | [x2paddle.models.wide_resnet50_2_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L400)  |
| [torchvision.models.wide_resnet101_2](https://pytorch.org/vision/stable/models.html#torchvision.models.wide_resnet101_2) | [x2paddle.models.wide_resnet101_2_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/resnet.py#L419) |
| [torchvision.models.vgg11](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg11) | [x2paddle.models.vgg11_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/vgg.py#L117)            |
| [torchvision.models.vgg11_bn](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg11_bn) | [x2paddle.models.vgg11_bn_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/vgg.py#L128)         |
| [torchvision.models.vgg13](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg13) | [x2paddle.models.vgg13_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/vgg.py#L139)            |
| [torchvision.models.vgg13_bn](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg13_bn) | [x2paddle.models.vgg13_bn_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/vgg.py#L150)         |
| [torchvision.models.vgg16](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg16) | [x2paddle.models.vgg16_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/vgg.py#L161)            |
| [torchvision.models.vgg16_bn](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg16_bn) | [x2paddle.models.vgg16_bn_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/vgg.py#L172)         |
| [torchvision.models.vgg19](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg19) | [x2paddle.models.vgg19_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/vgg.py#L183)            |
| [torchvision.models.vgg19_bn](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg19_bn) | [x2paddle.models.vgg19_bn_pth](https://github.com/PaddlePaddle/X2Paddle/blob/develop/x2paddle/project_convertor/pytorch/models/vgg.py#L194)         |
