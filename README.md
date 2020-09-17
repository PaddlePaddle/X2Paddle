# X2Paddle
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/X2Paddle.svg)](https://github.com/PaddlePaddle/X2Paddle/releases)  
X2Paddle支持将其余深度学习框架训练得到的模型，转换至PaddlePaddle模型。  
X2Paddle is a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks.

## 转换模型库
X2Paddle在多个主流的CV模型上，测试过TensorFlow/Caffe/ONNX模型的转换，可以在[X2Paddle-Model-Zoo](x2paddle_model_zoo.md)查看我们的模型测试列表，可以在[OP-LIST](op_list.md)中查看目前X2Paddle支持的OP列表。如果你在新的模型上进行了测试转换，也欢迎继续补充该列表；如若无法转换，可通过ISSUE反馈给我们，我们会尽快跟进。

## 环境依赖

python == 2.7 | python >= 3.5  
paddlepaddle >= 1.8.0  

**按需安装以下依赖**  
tensorflow ： tensorflow == 1.14.0  
caffe ： 无  
onnx ： onnx >= 1.6.0

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
pip install x2paddle --index https://pypi.Python.org/simple/
```
## 使用方法
### TensorFlow
```
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```
### Caffe
```
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel --save_dir=pd_model
```
### ONNX
```
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model
```
### Paddle2ONNX
```
# 注意：paddle_infer_model_dir下需包含__model__和__params__两个文件
x2paddle --framework=paddle2onnx --model=paddle_infer_model_dir --save_dir=onnx_model
```
### 参数选项
| 参数 | |
|----------|--------------|
|--framework | 源模型类型 (tensorflow、caffe、onnx、paddle2onnx) |
|--prototxt | 当framework为caffe时，该参数指定caffe模型的proto文件路径 |
|--weight | 当framework为caffe时，该参数指定caffe模型的参数文件路径 |
|--save_dir | 指定转换后的模型保存目录路径 |
|--model | 当framework为tensorflow/onnx时，该参数指定tensorflow的pb模型文件或onnx模型路径 |
|--caffe_proto | **[可选]** 由caffe.proto编译成caffe_pb2.py文件的存放路径，当存在自定义Layer时使用，默认为None |
|--without_data_format_optimization | **[可选]** For TensorFlow, 当指定该参数为False时，打开NHWC->NCHW的优化，见[文档Q2](FAQ.md)，默认为True|
|--define_input_shape | **[可选]** For TensorFlow, 当指定该参数时，强制用户输入每个Placeholder的shape，见[文档Q2](FAQ.md) |
|--params_merge | **[可选]** 当指定该参数时，转换完成后，inference_model中的所有模型参数将合并保存为一个文件__params__ |
|--onnx_opset | **[可选]** 当framework为paddle2onnx时，该参数可设置转换为ONNX的OpSet版本，目前支持9、10、11，默认为10 |



## 使用转换后的模型
转换后的模型包括`model_with_code`和`inference_model`两个目录。  
`model_with_code`中保存了模型参数，和转换后的python模型代码  
`inference_model`中保存了序列化的模型结构和参数，可直接使用paddle的接口进行加载，见[load_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_guides/low_level/inference.html#api-guide-inference)

## 小工具
X2Paddle提供了工具解决如下问题，详见[tools/README.md](tools/README.md)
1. 检测模型是否在PaddleLite中支持  
2. 合并模型参数文件

## 相关文档
1. [X2Paddle使用过程中常见问题](FAQ.md)  
2. [如何导出TensorFlow的pb模型](export_tf_model.md)
3. [X2Paddle测试模型库](x2paddle_model_zoo.md)  
4. [PyTorch模型导出为ONNX模型](pytorch_to_onnx.md)
5. [X2Paddle内置的Caffe自定义层](caffe_custom_layer.md)

## 更新历史
2019.08.05  
1. 统一tensorflow/caffe/onnx模型转换代码和对外接口
2. 解决上一版caffe2fluid无法转换多分支模型的问题
3. 解决Windows上保存模型无法加载的问题
4. 新增optimizer，优化代码结构，合并conv、batch_norm的bias和激活函数  

**如果你需要之前版本的tensorflow2fluid/caffe2fluid/onnx2fluid，可以继续访问release-0.3分支，获取之前版本的代码使用。**


## Acknowledgements

X2Paddle refers to the following projects:
- [MMdnn](https://github.com/microsoft/MMdnn)
