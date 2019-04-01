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
python -m onnx2fluid -e -o /path/to/export/model /path/of/onnx/model
```
### VGG19转换
```shell
# 下载并解压onnx模型vgg19
wget https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz
tar xzvf vgg19.tar.gz

# 转换为PaddlePaddle模型
python -m onnx2fluid -e -o paddle_model vgg19/model.onnx
```
