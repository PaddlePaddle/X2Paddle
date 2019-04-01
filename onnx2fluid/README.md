# onnx2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


onnx2fluid支持将onnx模型转换为PaddlePaddle模型，并用于预测。

## 环境安装

工具开发过程中，我们在如下环境配置中测试模型转换，建议使用[anaconda](https://docs.anaconda.com/anaconda/install)

> python2 & python3

> onnx == 1.4.0

> paddlepaddle == 1.3.0

``` shell
# 安装onnx
# 也可参考https://github.com/onnx/onnx
conda install -c conda-forge onnx
```

## 使用说明
         
```shell
# 运行目录 X2Paddle/onnx2fluid
python -m onnx2fluid -e -o /path/to/export/model /path/of/onnx/model

# 按如下流程安装后，则不限定上述命令的运行目录
python setup.py install
```
**VGG19转换**
```shell
# 下载并解压onnx模型vgg19
wget https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz
tar xzvf vgg19.tar.gz

# 转换为PaddlePaddle模型
python -m onnx2fluid -e -o paddle_model vgg19/model.onnx
```
转换后的PaddlePaddle模型加载可参考文档[加载预测模型](http://www.paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/inference.html#id4)

## 模型测试
onnx2fluid在如下模型中进行了测试
[bvlc_alexnet](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz)
[bvlc_googlenet](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz)
[bvlc_reference_caffenet](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz)
[bvlc_reference_rcnn_ilsvrc13](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_rcnn_ilsvrc13.tar.gz)
[inception_v1](https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v1.tar.gz)
[inception_v2](https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz)
[resnet50](https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz)
[shufflenet](https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz)
[squeezenet](https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz)
[vgg19](https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz)
[zfnet512](https://s3.amazonaws.com/download.onnx/models/opset_9/zfnet512.tar.gz)
