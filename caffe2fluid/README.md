# caffe2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

caffe2fluid用于将Caffe模型转换为PaddlePaddle模型

## 环境依赖

> python >= 2.7  
> numpy  
> protobuf >= 3.7.1  
> future  

**caffe2fluid的运行仅依赖上述条件**  
但建议在环境中安装好caffe和paddlepaddle，便于转换模型后测试。环境安装可参考[安装文档](prepare.md)

## 使用方法

### 模型转换
1. Caffe模型转换为PaddlePaddle模型代码和参数文件（参数以numpy形式保存）

```
# alexnet.prototxt : caffe配置文件
# --def_path : caffe配置文件的保存路径
# --caffemodel : caffe模型的保存路径
# --data-output-path : 转换后模型参数保存路径
# --code-output-path : 转换后模型代码保存路径
python convert.py --def_path alexnet.prototxt \
		--caffemodel alexnet.caffemodel \
		--data-output-path alexnet.npy \
		--code-output-path alexnet.py
```

2. 可通过如下方式，将模型网络结构和参数均序列化保存为PaddlePaddle框架支持加载的模型格式
```
# --model-param-path ： 指定序列化后的模型保存路径
python alexnet.py --npy_path alexnet.npy --model-param-path ./fluid_model
```
或者也可在保存时，指定保存模型的输出
```
# 模型的输出为fc8和prob层
python alexnet.py --npy_path alexnet.npy --model-param-path ./fluid --need-layers-name fc8,prob
```
模型的加载及预测可参考PaddlePaddle官方文档[加载预测模型](http://www.paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/inference.html#id4)

### 模型转换前后差异对比
模型转换后，可通过如下方式，逐层对比转换后的模型与原模型的计算结果差异（**运行环境依赖caffe和paddlepaddle**）
```
# alexnet : caffe配置文件（.prototxt）中“name”的值
# ./models/alexnet.prototxt : caffe配置文件路径
# ./models/alexnet.caffemodel : caffe模型文件路径
# ./models/alexnet.py : 转换后模型代码保存路径
# ./models/alexnet.npy : 转换后模型参数保存路径
# ./data/65.jpeg : 需要测试的图像数据
cd examples/imagenet
bash tools/diff.sh alexnet ./models/alexnet.prototxt \
			./models/alexnet.caffemodel \
			./models/alexnet.py \
			./models/alexnet.npy \
			./data/65.jpeg
```

## 自定义层转换
在模型转换中遇到未支持的自定义层，用户可根据自己需要，添加代码实现自定义层，从而支持模型的完整转换，实现方式如下流程，
1. 在`kaffe/custom_layers`下实现自定义层，例如mylayer.py
> - 实现`shape_func(input_shape, [other_caffe_params])`，计算输出的大小
> - 实现`layer_func(input_shape, [other_caffe_params])`，构造一个PaddlePaddle Fluid层
> - 注册这两个函数 `register(kind=`MyType`, shape=shape_func, layer=layer_func)`
也可参考`kaffe/cusom_layers`下的其它自定义层实现

2. 添加`import mylayer`至`kaffe/custom_layers/__init__.py`

3. 准备你的pycaffe作为你的定制版本（与以前的env准备相同）
> 选择一：编译你自己的caffe.proto来代替proto/caffe.proto  
> 选择二：更换你的pycaffe到特定的版本

4. 按照之前步骤，将Caffe模型转换为PaddlePaddle模型

5. 配置环境变量
```
export CAFFE2FLUID_CUSTOM_LAYERS=/path/to/caffe2fluid/kaffe
```
## 模型测试
caffe2fluid在如下模型上通过测试
- [Lenet](https://github.com/ethereon/caffe-tensorflow/blob/master/examples/mnist)
- [ResNet(ResNet-50,ResNet-101,ResNet-152)](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
- [GoogleNet](https://gist.github.com/jimmie33/7ea9f8ac0da259866b854460f4526034)
- [VGG](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)
- [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
