# onnx2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


onnx2fluid支持将onnx模型转换为PaddlePaddle模型，并用于预测。

## 环境安装

工具开发过程中，我们在如下环境配置中测试模型转换，建议使用[anaconda](https://docs.anaconda.com/anaconda/install)

> python2 & python3

> onnx == 1.12.0

> paddlepaddle == 1.3.0

``` shell
# 安装onnx
# 安装也可参考https://github.com/onnx/onnx
conda install -c conda-forge protobuf numpy
pip install onnx
```
         
