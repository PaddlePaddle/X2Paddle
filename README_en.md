# X2Paddle

[![PyPI - X2Paddle Version](https://img.shields.io/pypi/v/x2paddle.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/x2paddle/)
[![PyPI Status](https://pepy.tech/badge/x2paddle/month)](https://pepy.tech/project/x2paddle)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/X2Paddle.svg)](https://github.com/PaddlePaddle/X2Paddle/releases)
![python version](https://img.shields.io/badge/python-3.8+-orange.svg)

## Introduction

X2Paddle is a model conversion tool under the PaddlePaddle ecosystem, dedicated to helping users of other deep learning frameworks to quickly migrate to PaddlePaddle framework. Currently supports **inference model conversion** and **PyTorch training code migration**. We also provide a detailed API comparison document between different frameworks, to reduce the time and cost of developers to migrate their models to PaddlePaddle.

## Features

- **Supports major deep learning frameworks**

  - It has supported the conversion on inference models of **Caffe/TensorFlow/ONNX/PyTorch** and the conversion of PyTorch training projects, which covers the major deep learning frameworks in the market at present. For further details, refer to ***[test demo ](./test_benchmark)***

- **Rich set of supported models**

  - Supports most model conversion on mainstream CV and NLP models. Currently X2Paddle supports 130+ PyTorch OPs, 90+ ONNX OPs, 90+ TensorFlow OPs and 30+ Caffe OPs. For further details, refer to ***[support list](./docs/inference_model_convertor/op_list.md)***

- **Simple and easy to use**

  - Model conversion can be done with a single command or an API.

## Capabilities

- **Inference model conversion**

  - Support one-stop model conversion from Caffe/TensorFlow/ONNX/PyTorch to PaddlePaddle inference model, and use PaddleInference/PaddleLite for CPU/GPU/Arm and other devices deployment.

- **PyTorch training projects conversion**

  - Support one-stop conversion of PyTorch project Python code (including training, prediction) into the project based on PaddlePaddle framework, helping developers quickly migrate their project quickly. And can enjoy [AIStudio platform](https://aistudio.baidu.com/), which provide a large number of free computational power. **[New feature, try it!](/docs/pytorch_project_convertor/README.md)**

- **API Documentation**

  - Detailed comparative analysis of API documentation to help developers quickly migrate from the use of the PyTorch framework to the use of PaddlePaddle framework, greatly reducing the cost of learning. **[New content, learn about!](docs/pytorch_project_convertor/API_docs/README.md)**


## Installation

### Environment Dependencies

- python >= 3.8
- paddlepaddle >= 2.2.2 (Officially verified to `3.0.0beta1`)
- tensorflow == 1.14 (For TensorFlow model conversion. Where `test_benchmark` models have been tested on `2.16.1`.)
- onnx >= 1.6.0 (For ONNX model conversion. Where `test_benchmark` models have been tested on `1.17.0`.)
- torch >= 1.5.0 (For PyTorch model conversion. Where `test_benchmark` models have been tested on `2.4.1`.)
- paddlelite >= 2.9.0 (For conversion to Paddle-Lite supported formats, the latest version is recommended.)

> Note: The above tested versions do not mean that X2Paddle supports all the operators in the corresponding version, it only means that they can be used in this environment. Please refer to [support List](./docs/inference_model_convertor/op_list.md)

### pip Installation (Recommended)

For a stable version, install X2Paddle via pip:

```
pip install x2paddle
```

### Source Code Installation

If you want to experience the latest features, you can use the source code installation method:

```
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
git checkout develop
python setup.py install
```

## Quick Start

### Feature 1: Inference Model Conversion

#### PyTorch model Conversion

``` python
from x2paddle.convert import pytorch2paddle
pytorch2paddle(module=torch_module,
               save_dir="./pd_model",
               jit_type="trace",
               input_examples=[torch_input])
# module (torch.nn.Module): PyTorch's Module
# save_dir (str): Save path of inference model
# jit_type (str): Convert approach. Default is "trace"
# input_examples (list[torch.tensor]): torch.nn.Module's input. The length of list must be equal to the input's. Default is None.
```

The ```script``` approach more details can be found [PyTorch model conversion documentation](./docs/inference_model_convertor/pytorch2paddle.md).

#### TensorFlow Model Conversion

```shell
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```

#### ONNX Model Conversion

```shell
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model
```

#### Caffe Model Conversion

```shell
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel --save_dir=pd_model
```

#### Conversion Parameter Description

| Parameters           | Description                                                         |
| -------------------- | ------------------------------------------------------------ |
| --framework          | Source model type (TensorFlow, Caffe, ONNX) |
| --prototxt           | When the framework is Caffe, this parameter specifies the path to the proto file of the caffe model |
| --weight             | When the framework is Caffe, this parameter specifies the path to the parameter file for the caffe model |
| --save_dir           | Specify the path to the directory where the converted model is saved |
| --model              | When framework is TensorFlow/ONNX, this parameter specifies TensorFlow's pb file path or ONNX's model file path |
| --input_shape_dict   | **[optional]** For ONNX, ONNX input shape |
| --caffe_proto        | **[optional]** Path to the caffe_pb2.py file compiled from caffe.proto, used when a custom Layer exists, default is None |
| --define_input_shape | **[optional]** For TensorFlow, When this parameter is specified, forces the user to enter the shape of each Placeholder, see [documentation Q2](./docs/inference_model_convertor/FAQ.md) |
| --enable_code_optim  | **[optional]** For PyTorch, Whether to optimize the generated code, default is False |
| --to_lite            | **[optional]** Whether to use the `opt` tool to convert to Paddle-Lite supported formats, default is False |
| --lite_valid_places  | **[optional]** Specify the type of conversion, you can specify more than one backend at the same time (separated by commas), opt will automatically select the best way, default is arm. |
| --lite_model_type    | **[optional]** Specify the model conversion type, currently supports two types: protobuf and naive_buffer, default is naive_buffer |
| --disable_feedback   | **[optional]** Whether or not to turn off X2Paddle feedback; By default, X2Paddle will count the success rate in model conversion, as well as the source of the conversion framework and other information, in order to help X2Paddle iterate according to the user's needs. X2Paddle will not upload the user's model files. If you don't want to participate in the feedback, you can specify this parameter as False. |

#### X2Paddle API

Currently X2Paddle provides API to convert models, you can refer to [X2PaddleAPI](docs/inference_model_convertor/x2paddle_api.md)

#### One-Stop Conversion of Paddle-Lite Supported Formats

Refer to [convert2lite_api](docs/inference_model_convertor/convert2lite_api.md)

### Feature 2: PyTorch Project Migration

Project conversion consists of 3 steps

1. project code preprocessing
2. one-stop code/pre-trained model conversion
3. post-processing of the converted code

Refer to [pytorch_project_convertor](./docs/pytorch_project_convertor/README.md)

### Model Conversion with VisualDL

VisualDL, the PaddlePaddle visualization tool, has deployed the model conversion tool on the official website to provide service, you can click [Service Link](https://www.paddlepaddle.org.cn/paddle/visualdl/modelconverter/) to perform online ONNX2Paddle model conversion.

![ONNX2Paddle](https://user-images.githubusercontent.com/22424850/226797893-ef697887-a056-445f-933e-f1bbc7c7df76.gif)

## Tutorials

1. [TensorFlow Inference Model Conversion Tutorial](./docs/inference_model_convertor/demo/tensorflow2paddle.ipynb)
2. [MMDetection Model Conversion Guide](./docs/inference_model_convertor/toolkits/MMDetection2paddle.md)
3. [PyTorch Inference Model Conversion Tutorial](./docs/inference_model_convertor/demo/pytorch2paddle.ipynb)
4. [PyTorch Training Project Conversion Tutorial](./docs/pytorch_project_convertor/demo/README.md)

## :hugs:Contributing:hugs:

You are more than welcome to contribute code to X2Paddle or provide suggestions on how to use it.
- If you can fix an issue or add a new feature, feel free to send us Pull Requests!
- You can use the [development mirror](./docker)：[paddlepaddle/x2paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82](https://hub.docker.com/r/paddlepaddle/x2paddle/tags)
- Please feel free to submit an issue if you need to convert your PyTorch training program.
