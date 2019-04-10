# X2Paddle
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

# 简介

X2Paddle支持将Caffe和TensorFlow模型转至PaddlePaddle模型，同时我们目前维护了TensorFlow/Caffe与PaddlePaddle接口对比分析文档。  
任何使用问题均可通过ISSUE的方式及时反馈，或者也可直接通过pull request的方式一起更新代码和文档。

## [caffe2fluid](caffe2fluid)
1. 支持将Caffe模型转至PaddlePaddle fluid可加载预测模型
2. 提供Caffe-PaddlePaddle常用API的对比文档[[doc](caffe2fluid/doc)]

## [tensorflow2fluid](tensorflow2fluid)
1. 支持将TensorFlow模型转至PaddlePaddle fluid可加载预测模型
2. 提供TensorFlow-PaddlePaddle常用API的对比文档[[doc](tensorflow2fluid/doc)]

## [onnx2fluid](onnx2fluid)
1. 支持将ONNX模型转至PaddlePaddle fluid可加载预测模型
2. Pytorch支持导出为ONNX模型，因此也可通过onnx2fluid支持PyTorch模型的转换

# 贡献代码
clone代码至本地后，先运行`X2Paddle/commit-prepare.sh`配置代码提交环境
