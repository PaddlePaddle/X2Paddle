# onnx2fluid

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

onnx2fluid supports converting ONNX model to PaddlePaddle fluid model for prediction.

PyTorch to Paddlepaddle model conversion can be easily achieved with PyTorch ONNX export functions.

## Features

* Python code + ProgramDesc proto generation, flexible and compatible
* fluid layer weight embedding support
* conversion, validation, archiving all in one
* convert without PaddlePaddle dependency
* export and validation helper functions for PyTorch to PaddlePaddle conversion
* extra ONNX operator optimization for inference
* easily extensible for user-defined operators

## Environment and dependency

* python 3.5+ (python 2 not fully supported yet)
* onnx >= 1.4
* paddlepaddle >= 1.3.0 (optional for validation)

## Get started

Test with pretrained models from ONNX repositories:

``` shell
python setup.py install
cd examples
sh onnx_model_zoo.sh
```

Try exporting and validating from PyTorch to PaddlePaddle fluid:

``` shell
python setup.py install
cd examples

python gen_some_samples.py
onnx2fluid sample_1.onnx -t sample_1.npz

python gen_unet.py
onnx2fluid sample_unet.onnx -t sample_unet.npz
```

## Usage

**ONNX opset 9+** is mainly supported, corresponded to PyTorch **1.0/1.1(stable opset)**，for more information: [ONNX doc](https://github.com/onnx/onnx/blob/master/docs/Operators.md)

onnx2fluid (all in one):

```shell
onnx2fluid [-dexy] [-o /path/to/export_dir/] [-z archive.zip] [-t test_data.npz] [-i [input_name1,input_name2]] /path/to/onnx/model.onnx

optional arguments:
  --debug, -d           enable debug logging and checking
  --embed_params, -e    try to embed parameters for trainable PaddlePaddle fluid layers
  --no-pedantic, -x     process non-standard ONNX ops
  --skip-version-conversion, -y
                        skip ONNX op version conversion, workaround for RumtimeErrors
  --output_dir, -o      output directory
  --archive [ARCHIVE], -z [ARCHIVE]
                        compress outputs to ZIP file if conversion successed
  --infer_inputs, -i [input_name1,input_name2]
                        invoke PaddlePaddle fluid type-shape inference
```

onnx2fluid.conversion:

```shell
onnx2fluid.conversion [-dexy] [-o /path/to/export_dir/] /path/to/onnx/model.onnx
```

onnx2fluid.validate:

```shell
onnx2fluid.validate [-d] [-t test_data.npz] [-i [input_name1,input_name2]] [-p 1e-3] /path/to/onnx/model.onnx
```

## Reference

* [PaddlePaddle fluid operators](http://www.paddlepaddle.org/documentation/docs/en/1.5/api/layers.html)
* load converted model via [load_inference_model](http://www.paddlepaddle.org/documentation/docs/en/1.5/api/io.html#permalink-1-load_inference_model)
