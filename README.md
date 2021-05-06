# X2Paddle
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/X2Paddle.svg)](https://github.com/PaddlePaddle/X2Paddle/releases)
![python version](https://img.shields.io/badge/python-3.5+-orange.svg)  

## 简介
X2Paddle用于不同框架模型或项目到PaddlePaddle框架模型或项目的迁移，旨在为飞桨开发者提升框架间迁移的效率。

### 特性
- **支持主流深度学习框架**：目前已经支持Caffe/TensorFlow/ONNX/PyTorch四大框架的迁移，涵盖目前市面主流深度学习框架。
- **支持的模型丰富**：在主流的CV和NLP模型上均支持转换，涵盖了19+个Caffe模型转换、27+个TensorFlow模型转换、32+个ONNX模型转换、27+个PyTorch模型转换、2+个PyTorch项目转换。
- **简洁易用**：一条命令行或者一个API即可完成模型转换。

### X2Paddle技术文档
- [各框架OP算子支持列表](docs/introduction/op_list.md)
- [各框架模型转换支持列表](docs/introduction/x2paddle_model_zoo.md)
- [X2Paddle技术原理](docs/introduction/architecture.md)

## 安装

### 环境依赖

- python >= 3.5  
- paddlepaddle >= 2.0.0

- 转换TensorFlow模型时: tensorflow==1.14.0
- 转换ONNX模型时: onnx >= 1.6.0
- 转换PyTorch模型时: torch >= 1.5.0

### 安装方式

#### 1. pip安装(推荐)
```
pip install x2paddle
```

#### 2. 源码安装
```
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
git checkout develop
python setup.py install
```

## 快速开始
### Caffe模型转换
```
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel --save_dir=pd_model
```

### TensorFlow模型转换
```
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```

### ONNX模型转换
```
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model --paddle_type dygraph
```

### PyTorch模型转换
PyTorch模型目前不支持命令方式，需用户通过Python API的方式调用转换，详见[PyTorch模型转换](docs/user_guides/pytorch2paddle.md)

### 转换命令参数说明

| 参数                 |        作用                                                      |
| -------------------- | ------------------------------------------------------------ |
| --framework          | 源模型类型 (tensorflow、caffe、onnx)                         |
| --prototxt           | 当framework为caffe时，该参数指定caffe模型的proto文件路径     |
| --weight             | 当framework为caffe时，该参数指定caffe模型的参数文件路径      |
| --save_dir           | 指定转换后的模型保存目录路径                                 |
| --model              | 当framework为tensorflow/onnx时，该参数指定tensorflow的pb模型文件或onnx模型路径 |
| --caffe_proto        | **[可选]** 由caffe.proto编译成caffe_pb2.py文件的存放路径，当存在自定义Layer时使用，默认为None |
| --define_input_shape | **[可选]** For TensorFlow, 当指定该参数时，强制用户输入每个Placeholder的shape，见[文档Q2](./docs/user_guides/FAQ.md) |
| --paddle_type        | **[可选]** 该参数指定转换为动态图代码（dygraph）或者静态图代码（static），默认为dygraph |


## 其它相关文档
1. [X2Paddle使用过程中常见问题](./docs/user_guides/FAQ.md)  
2. [如何导出TensorFlow的Frozen Model](./docs/user_guides/export_tf_model.md)
3. [PyTorch模型导出为ONNX模型](./docs/user_guides/pytorch2onnx.md)
4. [X2Paddle添加内置的Caffe自定义层](./docs/user_guides/add_caffe_custom_layer.md)
5. [转换后PaddlePaddle预测模型简介](./docs/user_guides/pd_folder_introduction.py)
6. [Paddle到ONNX的转换](https://github.com/PaddlePaddle/Paddle2ONNX)
7. [X2Paddle测试模型库](./docs/introduction/x2paddle_model_zoo.md)  
8. [X2Paddle支持的op列表](./docs/introduction/op_list.md)


## 转换教程
1. [TensorFlow预测模型转换教程](./docs/demo/tensorflow2paddle.ipynb)
2. [PyTorch预测模型转换教程](./docs/demo/pytorch2paddle.ipynb)

## 更新历史
2020.12.09  
1. 新增PyTorch2Paddle转换方式，转换得到Paddle动态图代码，并动转静获得inference_model。  
  方式一：trace方式，转换后的代码有模块划分，每个模块的功能与PyTorch相同。    
  方式二：script方式，转换后的代码按执行顺序逐行出现。  
2. 新增Caffe/ONNX/Tensorflow到Paddle动态图的转换。
3. 新增TensorFlow op映射（14个）：Neg、Greater、FloorMod、LogicalAdd、Prd、Equal、Conv3D、Ceil、AddN、DivNoNan、Where、MirrorPad、Size、TopKv2。
4. 新增Optimizer模块，主要包括op融合、op消除功能，转换后的代码可读性更强，进行预测时耗时更短。

2021.04.30
1. 新增支持转换的模型：[SwinTransformer](https://github.com/microsoft/Swin-Transformer/)、[BASNet](https://github.com/xuebinqin/BASNet)、[DBFace](https://github.com/dlunion/DBFace)、[EasyOCR](https://github.com/JaidedAI/EasyOCR)、[CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)等。
2. 支持Windows上使用本工具。
3. 新增TensorFlow op映射（4个）：SplitV、ReverseV2、BatchToSpaceND、SpaceToBatchND。
4. 新增PyTorch op映射（11个）：aten::index、aten::roll、aten::adaptive_avg_pool1d、aten::reflection_pad2d、aten::reflection_pad1d、aten::instance_norm、aten::gru、aten::norm、aten::clamp_min、aten:prelu、aten:split_with_sizes。
5. 新增ONNX op映射（1个）：DepthToSpace。
6. 新增Caffe op映射（1个）：op：MemoryData。

## 贡献代码

我们非常欢迎您为X2Paddle贡献代码或者提供使用建议。如果您可以修复某个issue或者增加一个新功能，欢迎给我们提交Pull Requests。
