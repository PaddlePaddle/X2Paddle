# onnx2fluid

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

onnx2fluid支持将onnx模型转换为PaddlePaddle模型，并用于预测，用户也可以通过将Pytorch模型导出为ONNX格式模型，再使用onnx2fluid将模型转为PaddlePaddle模型。

## 环境安装

工具开发过程中，我们在如下环境配置中测试模型转换:

* python3.5+
* onnx == 1.4.0
* paddlepaddle == 1.3.0

建议使用[anaconda](https://docs.anaconda.com/anaconda/install):

``` shell
# 安装onnx
# 也可参考https://github.com/onnx/onnx
conda install -c conda-forge onnx
```

## 使用说明
```shell
# 安装
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle/onnx2fluid
python setup.py install

# 模型转换
python -m onnx2fluid -o /path/to/export_dir/ /path/of/onnx/model.onnx
```
**示例：VGG19模型**
```shell
wget https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz
tar xzvf vgg19.tar.gz

python -m onnx2fluid -o paddle_model vgg19/model.onnx
```
转换后的PaddlePaddle模型加载可参考文档[加载预测模型](http://www.paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/inference.html#id4)

## 模型测试
目录[examples](examples)中集成了部分ONNX预训练模型的转换测试
```shell
cd examples
# 测试和验证各onnx模型的转换
sh onnx_model_zoo.sh
```
目前测试脚本中已包含的测试模型如下，  
1. [bvlc_alexnet](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz)  
2. [bvlc_googlenet](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz)  
3. [bvlc_reference_caffenet](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz)  
4. [bvlc_reference_rcnn_ilsvrc13](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_rcnn_ilsvrc13.tar.gz)  
5. [inception_v1](https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v1.tar.gz)  
6. [inception_v2](https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz)  
7. [resnet50](https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz)  
8. [shufflenet](https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz)  
9. [squeezenet](https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz)  
10. [vgg19](https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz)  
11. [zfnet512](https://s3.amazonaws.com/download.onnx/models/opset_9/zfnet512.tar.gz)
