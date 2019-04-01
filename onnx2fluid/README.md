# onnx2fluid

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

onnx2fluid支持将onnx模型转换为PaddlePaddle模型，并用于预测，用户也可以通过将Pytorch模型导出为ONNX格式模型，再使用onnx2fluid将模型转为PaddlePaddle模型。

## 环境安装

工具开发过程中，我们在如下环境配置中测试模型转换:

* python3.5+ (python2 working in progress)
* onnx == 1.4.0
* paddlepaddle == 1.3.0

建议使用[anaconda](https://docs.anaconda.com/anaconda/install):

``` shell
# 安装onnx
# 也可参考https://github.com/onnx/onnx
conda install -c conda-forge onnx
```

## Get started

Test with pretrained models from ONNX repositories:

``` shell
python setup.py install
cd examples
sh onnx_model_zoo.sh
```

Try exporting from PyTorch to Paddle fluid:

``` shell
python setup.py install
cd examples
python gen_some_samples.py
onnx2fluid sample_1.onnx -t sample_1.npz
```

## 使用说明

```shell
onnx2fluid [-dexy] -o /path/to/export_dir/ /path/of/onnx/model.onnx

optional arguments:
  --embed_params, -e    try to embed parameters for trainable Paddle fluid layers
  --no-pedantic, -x     process non-standard ONNX ops
  --skip-version-conversion, -y
                        skip ONNX op version conversion, workaround for
                        RumtimeErrors
  --archive [ARCHIVE], -z [ARCHIVE]
                        compress outputs to ZIP file if conversion successed
```

转换后的PaddlePaddle模型加载可参考文档[加载预测模型](http://www.paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/inference.html#id4)
