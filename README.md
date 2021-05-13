# X2Paddle
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/X2Paddle.svg)](https://github.com/PaddlePaddle/X2Paddle/releases)
![python version](https://img.shields.io/badge/python-3.5+-orange.svg)  

## 简介
X2Paddle用于不同框架模型或项目到PaddlePaddle框架模型或项目的转换，旨在为飞桨开发者提升框架间转换的效率。  
X2Paddle主要有***2大功能***：  
1. ***预测模型转换***：X2Paddle支持Caffe/TensorFlow/ONNX/PyTorch的预测模型，一步转换至PaddlePaddle预测模型。
2. ***训练项目转换***：PyTorch训练项目，转换至PaddlePaddle项目，助力用户在PaddlePaddlePaddle上进行模型训练。

### 特性

- **支持主流深度学习框架**：目前已经支持Caffe/TensorFlow/ONNX/PyTorch四大框架的预测模型的转换，PyTorch训练项目的转换，涵盖了目前市面主流深度学习框架。  

- **支持的模型丰富**：在主流的CV和NLP模型上均支持转换，涵盖了19+个Caffe预测模型转换、27+个TensorFlow预测模型转换、32+个ONNX预测模型转换、27+个PyTorch预测模型转换、2+个PyTorch训练项目转换，详见 ***[支持列表](./docs/introduction/x2paddle_model_zoo.md)***。  

- **简洁易用**：一条命令行或者一个API即可完成模型转换。  



## 环境依赖

- python >= 3.5  
- paddlepaddle >= 2.0.0

**按需安装以下依赖**  
- tensorflow ： tensorflow == 1.14.0  
- caffe ： 无  
- onnx ： onnx >= 1.6.0  
- pytorch：torch >=1.5.0 (预测模型转换中的script方式暂不支持1.7.0+)

## 安装
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
| 参数                 |        作用                                                      |
| -------------------- | ------------------------------------------------------------ |
| --framework          | 源模型类型 (tensorflow、caffe、onnx)                         |
| --prototxt           | 当framework为caffe时，该参数指定caffe模型的proto文件路径     |
| --weight             | 当framework为caffe时，该参数指定caffe模型的参数文件路径      |
| --save_dir           | 指定转换后的模型保存目录路径                                 |
| --model              | 当framework为tensorflow/onnx时，该参数指定tensorflow的pb模型文件或onnx模型路径 |
| --caffe_proto        | **[可选]** 由caffe.proto编译成caffe_pb2.py文件的存放路径，当存在自定义Layer时使用，默认为None |
| --define_input_shape | **[可选]** For TensorFlow, 当指定该参数时，强制用户输入每个Placeholder的shape，见[文档Q2](./docs/inference_model_convertor/FAQ.md) |

#### TensorFlow
```shell
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```
【注意】目前只支持FrozenModel格式的TensorFlow模型到PaddlePaddle模型的转换，若为checkpoint或者SavedModel格式的TensorFlow模型参见[文档](./docs/inference_model_convertor/export_tf_model.md)导出FrozenModel格式模型。
#### Caffe
```shell
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel --save_dir=pd_model
```
【注意】若caffe模型中出现自定义层，需要按照[相关流程](./docs/inference_model_convertor/add_caffe_custom_layer.md)自行添加自定义层的转换代码。
#### ONNX
```shell
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model
```
【注意】如若需要将PyTorch模型转换为ONNX模型，可参见[PyTorch2ONNX转换文档](./docs/inference_model_convertor/pytorch2onnx.md)。
#### PyTorch
PyTorch仅支持API使用方式，详见[PyTorch预测模型转换文档](./docs/inference_model_convertor/pytorch2paddle.md)。  

***[预测模型转换常见问题](./docs/inference_model_convertor/FAQ.md)***


### 功能二：训练项目转换

| 参数 | 作用 |
|----------|--------------|
|--convert_torch_project | 当前方式为对PyTorch Project进行转换 |
|--project_dir | PyTorch的项目路径 |
|--save_dir | 指定转换后项目的保存路径 |
|--pretrain_model | **[可选]**需要转换的预训练模型的路径(文件后缀名为“.pth”、“.pt”、“.ckpt”)或者包含预训练模型的文件夹路径，转换后的模型将将保在当前路径，后缀名为“.pdiparams” |

```shell
x2paddle --convert_torch_project --project_dir=torch_project --save_dir=paddle_project --pretrain_model=model.pth
```
【注意】需要搭配预处理和后处理一起使用，详细可参见[训练项目转换文档](./docs/pytorch_project_convertor/README.md)。  

***[训练项目转换常见问题](./docs/pytorch_project_convertor/FAQ.md)***


## 转换教程
1. [TensorFlow预测模型转换教程](./docs/inference_model_convertor/demo/tensorflow2paddle.ipynb)
2. [PyTorch预测模型转换教程](./docs/inference_model_convertor/demo/pytorch2paddle.ipynb)
3. [PyTorch训练项目转换教程](./docs/pytorch_project_convertor/demo.md)

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
4. 新增PyTorch op映射（11个）：aten::index、aten::roll、aten::adaptive_avg_pool1d、aten::reflection_pad2d、aten::reflection_pad1d、aten::instance_norm、aten::gru、aten::norm、aten::clamp_min、aten::prelu、aten:split_with_sizes。
5. 新增ONNX op映射（1个）：DepthToSpace。
6. 新增Caffe op映射（1个）：MemoryData。

2021.05.13  
- 新增PyTorch训练项目功能：
支持转换的项目有[StarGAN](https://github.com/yunjey/stargan)、[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)。


## 贡献代码

我们非常欢迎您为X2Paddle贡献代码或者提供使用建议。如果您可以修复某个issue或者增加一个新功能，欢迎给我们提交Pull Requests。
