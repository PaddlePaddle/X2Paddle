### X2Paddle已经发布0.5版本，提供统一的模型转换工具，支持onnx/caffe/tensorflow，最新代码在develop分支：）
develop分支： https://github.com/PaddlePaddle/X2Paddle/tree/develop

# onnx2fluid

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

onnx2fluid支持将ONNX模型转换为PaddlePaddle模型，并用于预测，用户也可以通过将PyTorch模型导出为ONNX模型，再使用onnx2fluid将模型转为PaddlePaddle模型。

## 特色

* 导出Python代码和fluid ProgramDesc模型
* 权重可嵌入支持的算子中
* 转换验证打包三合一
* 转换过程不依赖PaddlePaddle
* 可自由扩展算子

## 环境配置

在如下环境配置中测试成功:

* python 3.5+
* onnx == 1.4.1
* paddlepaddle == 1.5.0 (可选，仅用于验证)

使用[Anaconda](https://docs.anaconda.com/anaconda/install):
``` shell
conda install -c conda-forge onnx
pip install paddlepaddle==1.5.0
```

## 动手玩

测试ONNX官方预训练模型，包含alexnet, googlenet, caffenet, rcnn
inception_v1, inception_v2, resnet50, shufflenet, squeezenet,
vgg19, zfnet512等:

``` shell
python setup.py install
cd examples
sh onnx_model_zoo.sh
```

使用PyTorch搭建模型，导出ONNX，转换并验证:

``` shell
python setup.py install
cd examples
python gen_some_samples.py
onnx2fluid sample_1.onnx -t sample_1.npz
```

## 使用说明

目前支持 **ONNX opset 9+** 的部分算子，对应PyTorch版本 **1.0/1.1(stable opset)**，更多兼容信息请参考[ONNX文档](https://github.com/onnx/onnx/blob/master/docs/Operators.md)

onnx2fluid:

```shell
onnx2fluid [-dexy] [-o /path/to/export_dir/] [-z archive.zip] [-t test_data.npz] [-i [input_name1,input_name2]] /path/to/onnx/model.onnx

optional arguments:
  --debug, -d           启用调试
  --embed_params, -e    尝试权重内嵌
  --no-pedantic, -x     转换扩展的ONNX算子
  --skip-version-conversion, -y
                        跳过ONNX算子版本转换
  --output_dir, -o      指定输出目录
  --archive [ARCHIVE], -z [ARCHIVE]
                        如果验证通过，打包到指定的ZIP文件
  --infer_inputs, -i [input_name1,input_name2]
                        调用PaddlePaddle fluid类形推导完善模型
```

转换工具onnx2fluid.conversion:

```shell
onnx2fluid.conversion [-dexy] [-o /path/to/export_dir/] /path/to/onnx/model.onnx
```

验证工具onnx2fluid.validate:

```shell
onnx2fluid.validate [-d] [-t test_data.npz] [-i [input_name1,input_name2]] [-p 1e-3] /path/to/onnx/model.onnx
```

## 参考

* PaddlePaddle [算子](http://www.paddlepaddle.org/documentation/docs/zh/1.5/api_cn/layers_cn.html)
* PaddlePaddle [加载预测模型](http://www.paddlepaddle.org/documentation/docs/zh/1.5/api_guides/low_level/inference.html#id4)
