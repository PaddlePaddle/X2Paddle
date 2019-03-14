# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


tensorflow2fluid支持将训练好的TensorFlow模型转换为PaddlePaddle模型，包括基于PaddlePaddle实现的模型前向计算网络python代码，以及PaddlePaddle可加载的模型参数文件。
> <a href="#环境依赖">`环境依赖`</a>
> <a href="#使用方法">`使用方法`</a>
> <a href="#测试模型">`测试模型`</a>

**我们计划专门梳理出指南文档，对比TensorFlow与PaddlePaddle的差异，帮助TensorFlow用户快速上手PaddlePaddle的使用，文档后续会整理在doc目录下，欢迎有需求的同学关注！**

## 依赖环境

工具开发过程中，我们在如下环境配置中测试模型转换

> python == 2.7 or 3.6

> tensorflow == 1.12.0

> paddlepaddle == 1.3.0

<a id="使用方法">
         
## 使用方法
### 转换模型
```
python src/convert.py --pb_file tf_model.pb \
                      --in_nodes inputs \
                      --output_nodes outputs \
                      --input_shape None,224,224,3 \
                      --input_format NHWC \
                      --use_cuda True \
                      --save_dir translated_paddle_model
```
### 加载模型并预测  
本目录下提供了[model_loader.py](tf2fluid/model_loader.py)，可以辅助用户简单的加载模型和预测，和dump模型，用户可直接参考其实现  

``` python
# coding:utf-8
# 代码运行目录 X2Paddle/tensorflow2fluid
import sys
import tf2fluid.model_loader as ml

# 加载模型
model = ml.ModelLoader("translated_paddle_model", use_cuda=True)

# 随机生成数据用于模型预测
# 注意Paddle CV模型输入格式为NCHW ！！！
data = numpy.random.rand(5, 3, 224, 224).astype('float32')
results = model.inference(feed_dict={model.inputs[0]:data})

# 返回的results为list，元素为np.array
for res in results:
    print(res.shape)
```

使用转换后的模型主要注意，**模型转换后，计算结果与原模型存在一定精度的diff，因此务必检查模型转换前后，在输入同样的数据前提下,diff是否符合预期**  

### 序列化模型结构  
tensorflow2fluid转换后的模型结构以python代码定义形式供用户直观阅读或修改，如若需要将模型结构和参数均序列化存储，可以上面的示例代码中，调用如下代码即可，序列化的模型结构和参数如何加载可见PaddlePaddle使用文档中的[加载预测模型](http://www.paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/inference.html#id4)
``` python
model.save_inference_model("new_model_dir")
```

### 参数说明  
|tf2fluid参数|说明|
|-----------|-----------------------------------------------|
|meta_file|TensorFlow模型序列化后保存的meta文件|
|ckpt_dir|TensorFlow模型保存checkpoint目录|
|pb_file|Tensorflow保存的pb格式模型|
|in_nodes|输入tensor名，多个输入时以空格分隔|
|input_shape|输入tensor的shape(batch维度以None表示)，shape之间以空格分隔，shape内各维度以逗号分隔|
|input_format|输入数据格式，NHWC/NCHW/OTHER|
|output_nodes|输出tensor名，多个输出时以空格分隔|
|use_cuda|转换过程中是否使用GPU，默认True|
|save_dir|转换后的模型保存路径|

目前支持tensorflow保存的checkpoint模型和将参数及模型结构序列化存储的pb模型，前者须指定meta_file和ckpt_dir，后者则指定pb_file

### 转换后模型文件说明  
文件|作用
:------------------:|:-----------------------------------------------:
mymodel.py|基于PaddlePaddle实现的模型网络结构python代码
ref_name.info|my_model.py中各tensor与原TensorFlow模型中的tensor对应关系
const_\*/params_\*|转换后的模型参数文件
save_var.list|模型载入过程中的变量list

### 加载转换后的模型并用于预测
本目录下的[model_loader.py](tf2fluid/model_loader.py)可用于加载转换后的模型

使用转换后的模型主要注意，**模型转换后，计算结果与原模型存在一定精度的diff，因此务必检查模型转换前后，在输入同样的数据前提下,diff是否符合预期**


## Link

本目录下部分代码参考了MMdnn-Tensorflow，对此表示感谢！

[MMdnn-Tensorflow](https://github.com/Microsoft/MMdnn/tree/master/mmdnn/conversion/tensorflow)
