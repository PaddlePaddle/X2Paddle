# X2Paddle

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/X2Paddle.svg)](https://github.com/PaddlePaddle/X2Paddle/releases)
![python version](https://img.shields.io/badge/python-3.5+-orange.svg)  

## 简介

X2Paddle致力于帮助其它主流深度学习框架用户快速迁移至飞桨框架，并为此提供了预测模型/训练代码的一键转换工具，以及框架间的API对比文档，提升用户在开发过程中的迁移效率，降低框架间迁移的学习成本。



## 特性

- **支持主流深度学习框架**

  - 目前已经支持Caffe/TensorFlow/ONNX/PyTorch四大框架的预测模型的转换，PyTorch训练项目的转换，涵盖了目前市面主流深度学习框架
   
- **支持的模型丰富**

  - 在主流的CV和NLP模型上均支持转换，涵盖了19+个Caffe预测模型转换、27+个TensorFlow预测模型转换、32+个ONNX预测模型转换、27+个PyTorch预测模型转换、2+个PyTorch训练项目转换，详见 ***[支持列表](./docs/introduction/x2paddle_model_zoo.md)***
 
- **简洁易用**

  - 一条命令行或者一个API即可完成模型转换



## 能力

- **预测模型转换**

  - 支持Caffe/TensorFlow/ONNX/PyTorch的模型一键转为飞桨的预测模型，并使用PaddleInference/PaddleLite进行CPU/GPU/Arm等设备的部署

- **PyTorch训练项目转换**

  - 支持PyTorch项目Python代码（包括训练、预测）一键转为基于飞桨框架的项目代码，帮助开发者快速迁移项目，并可享受[AIStudio平台](https://aistudio.baidu.com/)对于飞桨框架提供的海量免费计算资源[**【新功能，试一下！】**](/docs/pytorch_project_convertor/README.md)

- **API对应文档**

  - 详细的API文档对比分析，帮助开发者快速从PyTorch框架的使用迁移至飞桨框架的使用，大大降低学习成本 [**【新内容，了解一下！】**](docs/pytorch_project_convertor/API_docs/README.md)

  

## 安装

- python >= 3.5  
- paddlepaddle >= 2.0.0
- tensorflow == 1.14.0 (仅在转换TensorFlow模型时需要)
- onnx >= 1.6.0 (仅在转换ONNX模型时需要)
- torch >= 1.5.0 (仅在转换PyTorch模型时需要)

### 方式一：源码安装

```
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
git checkout develop
python setup.py install
```

### 方式二：pip安装(推荐）

我们会定期更新pip源上的x2paddle版本

```
pip install x2paddle --index https://pypi.python.org/simple/
```

## 快速开始

### 功能一：预测模型转换

```shell
# TensorFlow模型转换
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
# Caffe模型转换
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel --save_dir=pd_model
# ONNX模型转换
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model
# PyTorch模型转换 目前不支持命令行形式转换，参考下面链接文档进行转换
https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/pytorch2paddle.md
```

#### 转换参数说明

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --framework          | 源模型类型 (tensorflow、caffe、onnx)                         |
| --prototxt           | 当framework为caffe时，该参数指定caffe模型的proto文件路径     |
| --weight             | 当framework为caffe时，该参数指定caffe模型的参数文件路径      |
| --save_dir           | 指定转换后的模型保存目录路径                                 |
| --model              | 当framework为tensorflow/onnx时，该参数指定tensorflow的pb模型文件或onnx模型路径 |
| --caffe_proto        | **[可选]** 由caffe.proto编译成caffe_pb2.py文件的存放路径，当存在自定义Layer时使用，默认为None |
| --define_input_shape | **[可选]** For TensorFlow, 当指定该参数时，强制用户输入每个Placeholder的shape，见[文档Q2](./docs/inference_model_convertor/FAQ.md) |

### 功能二：PyTorch训练项目转换

项目转换包括3个步骤

1. 项目代码预处理
2. 代码/预训练模型一键转换
3. 转换后代码后处理

详见[PyTorch训练项目转换文档](./docs/pytorch_project_convertor/README.md)。


## 使用教程

1. [TensorFlow预测模型转换教程](./docs/inference_model_convertor/demo/tensorflow2paddle.ipynb)
2. [PyTorch预测模型转换教程](./docs/inference_model_convertor/demo/pytorch2paddle.ipynb)
3. [PyTorch训练项目转换教程](./docs/pytorch_project_convertor/demo/README.md)

## 更新历史

**2020.12.09**  

1. 新增PyTorch2Paddle转换方式，转换得到Paddle动态图代码，并动转静获得inference_model。  
   方式一：trace方式，转换后的代码有模块划分，每个模块的功能与PyTorch相同。  
   方式二：script方式，转换后的代码按执行顺序逐行出现。  
2. 新增Caffe/ONNX/Tensorflow到Paddle动态图的转换。
3. 新增TensorFlow op映射（14个）：Neg、Greater、FloorMod、LogicalAdd、Prd、Equal、Conv3D、Ceil、AddN、DivNoNan、Where、MirrorPad、Size、TopKv2。
4. 新增Optimizer模块，主要包括op融合、op消除功能，转换后的代码可读性更强，进行预测时耗时更短。

**2021.04.30**

1. 新增支持转换的模型：[SwinTransformer](https://github.com/microsoft/Swin-Transformer/)、[BASNet](https://github.com/xuebinqin/BASNet)、[DBFace](https://github.com/dlunion/DBFace)、[EasyOCR](https://github.com/JaidedAI/EasyOCR)、[CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)等。
2. 支持Windows上使用本工具。
3. 新增TensorFlow op映射（4个）：SplitV、ReverseV2、BatchToSpaceND、SpaceToBatchND。
4. 新增PyTorch op映射（11个）：aten::index、aten::roll、aten::adaptive_avg_pool1d、aten::reflection_pad2d、aten::reflection_pad1d、aten::instance_norm、aten::gru、aten::norm、aten::clamp_min、aten::prelu、aten:split_with_sizes。
5. 新增ONNX op映射（1个）：DepthToSpace。
6. 新增Caffe op映射（1个）：MemoryData。

**2021.05.13**  

- 新增PyTorch训练项目功能：
  支持转换的项目有[StarGAN](https://github.com/yunjey/stargan)、[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)。


## :hugs:贡献代码:hugs:

我们非常欢迎您为X2Paddle贡献代码或者提供使用建议。如果您可以修复某个issue或者增加一个新功能，欢迎给我们提交Pull Requests，如果有PyTorch训练项目转换需求欢迎随时提issue~
