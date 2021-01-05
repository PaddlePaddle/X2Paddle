# X2Paddle
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/X2Paddle.svg)](https://github.com/PaddlePaddle/X2Paddle/releases)  
X2Paddle支持将其余深度学习框架训练得到的模型，转换至PaddlePaddle模型。  
X2Paddle is a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks.

## 转换模型库
X2Paddle在多个主流的CV模型上，测试过TensorFlow/Caffe/ONNX/PyTorch模型的转换，可以在[X2Paddle-Model-Zoo](./docs/introduction/x2paddle_model_zoo.md)查看我们的模型测试列表，可以在[OP-LIST](./docs/introduction/op_list.md)中查看目前X2Paddle支持的OP列表。如果你在新的模型上进行了测试转换，也欢迎继续补充该列表；如若无法转换，可通过ISSUE反馈给我们，我们会尽快跟进。

## 环境依赖

python == 2.7 | python >= 3.5  
paddlepaddle 2.0.0-rc1 或者 develop  

**按需安装以下依赖**  
tensorflow ： tensorflow == 1.14.0  
caffe ： 无  
onnx ： onnx >= 1.6.0
pytorch：torch >=1.5.0 (script方式暂不支持1.7.0)

## 安装
### 安装方式一（推荐）
```
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
git checkout develop
python setup.py install
```

### 安装方式二
我们会定期更新pip源上的x2paddle版本
```
pip install x2paddle==1.0.0rc0 --index https://pypi.Python.org/simple/
```
## 使用方法
### TensorFlow
```
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model --paddle_type dygraph
```
### Caffe
```
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel --save_dir=pd_model --paddle_type dygraph
```
### ONNX
```
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model --paddle_type dygraph
```

### PyTorch
> PyTorch不支持命令行使用方式，详见[PyTorch2Paddle](./docs/user_guides/pytorch2paddle.md)

### Paddle2ONNX
> Paddle2ONNX功能已迁移至新的github: https://github.com/PaddlePaddle/paddle2onnx, 欢迎大家去新的代码仓库查看详细介绍以及新功能。


### 参数选项
| 参数 | |
|----------|--------------|
|--framework | 源模型类型 (tensorflow、caffe、onnx) |
|--prototxt | 当framework为caffe时，该参数指定caffe模型的proto文件路径 |
|--weight | 当framework为caffe时，该参数指定caffe模型的参数文件路径 |
|--save_dir | 指定转换后的模型保存目录路径 |
|--model | 当framework为tensorflow/onnx时，该参数指定tensorflow的pb模型文件或onnx模型路径 |
|--caffe_proto | **[可选]** 由caffe.proto编译成caffe_pb2.py文件的存放路径，当存在自定义Layer时使用，默认为None |
|--define_input_shape | **[可选]** For TensorFlow, 当指定该参数时，强制用户输入每个Placeholder的shape，见[文档Q2](./docs/user_guides/FAQ.md) |
|--params_merge | **[可选]** 当指定该参数时，转换完成后，inference_model中的所有模型参数将合并保存为一个文件__params__ |
|--paddle_type | **[可选]** 该参数指定转换为动态图代码（dygraph）或者静态图代码（static），默认为dygraph|



## 使用转换后的模型
- 静态图：
转换后的模型包括`model_with_code`和`inference_model`两个目录。  
`model_with_code`中保存了模型参数，和转换后的python模型静态图代码。  
`inference_model`中保存了序列化的模型结构和参数，可直接使用paddle的接口进行加载，见[paddle.static.load_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/static/load_inference_model_cn.html#load-inference-model)。
- 动态图：
转换后的模型包括`model.pdparams`和`x2paddle_code.py`两个文件，以及`inference_model`一个目录。  
`model.pdparams`中保存了模型参数。  
`x2paddle_code.py`是转换后的python模型动态图代码。  
`inference_model`中保存了序列化的模型结构和参数，可直接使用paddle的接口进行加载，见[paddle.static.load_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/static/load_inference_model_cn.html#load-inference-model)。

## 小工具
X2Paddle提供了工具解决如下问题，详见[tools/README.md](tools/README.md)
1. 检测模型是否在PaddleLite中支持  
2. 合并模型参数文件

## 相关文档
1. [X2Paddle使用过程中常见问题](./docs/user_guides/FAQ.md)  
2. [如何导出TensorFlow的pb模型](./docs/user_guides/export_tf_model.md)
3. [X2Paddle测试模型库](./docs/introduction/x2paddle_model_zoo.md)  
4. [X2Paddle支持的op列表](./docs/introduction/op_list.md) 
5. [PyTorch模型导出为ONNX模型](./docs/user_guides/pytorch2onnx.md)
6. [X2Paddle添加内置的Caffe自定义层](./docs/user_guides/add_caffe_custom_layer.md)

## 更新历史
2020.12.09
1. 新增PyTorch2Paddle转换方式，转换得到Paddle动态图代码，并动转静获得inference_model。
方式一：trace方式，转换后的代码有模块划分，每个模块的功能与PyTorch相同。
方式二：script方式，转换后的代码按执行顺序逐行出现。
2. 新增Caffe/ONNX/Tensorflow到Paddle动态图的转换。
3. 新增TensorFlow op（14个）：Neg、Greater、FloorMod、LogicalAdd、Prd、Equal、Conv3D、Ceil、AddN、DivNoNan、Where、MirrorPad、Size、TopKv2
4. 新增Optimizer模块，主要包括op融合、op消除功能，转换后的代码可读性更强，进行预测时耗时更短。


## Acknowledgements

X2Paddle refers to the following projects:
- [MMdnn](https://github.com/microsoft/MMdnn)
