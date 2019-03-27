caffe2fluid用于将Caffe模型转换为PaddlePaddle模型

# 环境安装

> python2/python3  
> caffe-gpu   
> paddlepaddle == 1.3.0  

建议在环境中安装好caffe和paddlepaddle，便于转换模型后测试。环境安装可参考[安装文档](#prepare.md)

# 使用方法

## 模型转换
1. Caffe模型转换为PaddlePaddle模型代码和参数文件（参数以numpy形式保存）

```
# alexnet.prototxt : caffe模型配置文件
# --caffemodel : caffe保存模型的路径
# --data-output-path : 转换后模型参数保存路径
# --code-output-path : 转换后模型代码保存路径
python convert.py alexnet.prototxt --caffemodel alexnet.caffemodel \
				          --data-output-path alexnet.npy \
					  --code-output-path alexnet.py
```

2. 可通过如下方式，将模型网络结构和参数均序列化保存为PaddlePaddle框架支持加载的模型格式
```
# fluid_model ： 指定序列化后的模型保存路径
python convert.py alexnet.py alexnet.npy fluid_model
```
也可在保存时，指定保存模型的输出
```
# 模型的输出为fc8和prob层
python convert.py alexnet.py alexnet.npy fluid_model fc8,prob
```
模型的加载及预测可参考PaddlePaddle官方文档[加载预测模型](http://www.paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/inference.html#id4)

## 模型转换前后差异对比
模型转换后，可通过如下方式，逐层对比转换后的模型与原模型的计算结果差异（运行环境依赖caffe和paddlepaddle）
```

bash tools/diff.sh alexnet
```
## 要点
1. 将Caffe模型及其对应的网络结构代码转换为Fluid模型和代码。
2. 通过扩展此工具也可以支持Caffe的自定义图层转换。
3. `examples/imagenet/tools`中提供了工具可以用于对此Caffe和Fluid预测后输出结果的差异。
## 准备工作
该部分主要介绍了使用此工具所需的环境安装。[详情](https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/prepare.md)
## 如何使用
1. 如果你的python中没有`pycaffe`模块，需要在`./proto`中加入`caffepb`，有以下两种方法可以实现这一操作。
> ```shell
> # 从caffe.proto中生成pycaffe
> bash ./proto/compile.sh
> # 直接从github上下载
> cd proto/ && wget https://raw.githubusercontent.com/ethereon/caffe-tensorflow/master/kaffe/caffe/caffeb.py
> ```
2. 将Caffe模型转换为Fluid模型
> ```shell
> # 将Caffe的模型和prototxt文件存放于`models`文件夹下
> # 生成Fluid代码和模型文件
> python convert.py ./models/alexnet.prototxt --caffemodel ./models/alexnet.caffemodel --data-output-path ./models/alexnet.npy --code-output-path ./models/alexnet.py
> # 将权值参数保存为Fluid模型文件
> python ./models/alexnet.py ./models/alexnet.npy ./models/fluid
> # 获取AlexNet中fc8层和prob层的结果
> python ./models/alexnet.py ./models/alexnet.npy ./models/fluid fc8,prob
> ```
3. 转换后并进行预测和比较（此部分需要Caffe和PaddlePaddle框架支持）
> ```shell
> cd examples/imagenet
>
> # 假设通过前一个步骤已经获得`../../models/fluid/model`和`../../models/fluid/params`，则可以使用Fluid进行预测
> python infer.py infer ../../models/fluid/ data/65.jpeg
>
> # 同时进行转换和预测
> bash ./tools/run.sh alexnet ../../models/ ../../models
> # 其中第一个参数为命名，第二个参数为Caffe代码和模型的存放路径，第三个参数为Fluid代码和模型的存放路径
> # 注意，Caffe和Fluid代码和模型的命名必须相同，只是后缀不同
>
> # 计算Caffe输出和Fluid输出的差异
> bash ./tools/diff.sh alexnet ../../models/ ../../models
> # 其中第一个参数为命名，第二个参数为Caffe代码和模型的存放路径，第三个参数为Fluid代码和模型的存放路径
> # 注意，Caffe和Fluid代码和模型的命名必须相同，只是后缀不同
> ```
## 如何转换自定义层
1. 在`kaffe/custom_layers`实现自定义的层，例如：mylayer.py   
  -实现`shape_func(input_shape, [other_caffe_params])`来计算输出的大小   
	-实现`layer_func(input_shape, [other_caffe_params])`来构造一个Fluid层   
	-运用这两个功能`register(kind='MyType', shape=shape_func, layer=layer_func)`    
	-注意：更多的示例可以从`kaffe/custom_layers`中找到
2. 将`import mylayer`添加到`kaffe/custom_layers/\__\_init__.py`中  
3. 准备你的pycaffe作为你的定制版本（与以前的env准备相同）  
	-选择一：编译你自己的`caffe.proto`来代替`proto/caffe.proto`  
	-选择二：更换你的`pycaffe`到特定的版本  
4. 将Caffe模型转换为Fluid模型
5. 设置环境变量`$CAFFE2FLUID_CUSTOM_LAYERS`为`custom_layers`的父目录
> ```shell
> export CAFFE2FLUID_CUSTOM_LAYERS=/path/to/caffe2fluid/kaffe
> ```
6. 使用转换好的模型
## 可测试的模型
- [Lenet](https://github.com/ethereon/caffe-tensorflow/blob/master/examples/mnist)
- [ResNet(ResNet-50,ResNet-101,ResNet-152)](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
- [GoogleNet](https://gist.github.com/jimmie33/7ea9f8ac0da259866b854460f4526034)
- [VGG](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)
- [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)



