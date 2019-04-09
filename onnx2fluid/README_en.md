# onnx2fluid

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

onnx2fluid supports converting ONNX model to PaddlePaddle Model for prediction.

## Running Environment

* python 3.5+ (python 2 working in progress)
* onnx == 1.4.0
* paddlepaddle == 1.3.0

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

## Usage

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
