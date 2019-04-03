# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


tensorflow2fluid支持将训练好的TensorFlow模型转换为PaddlePaddle模型，包括基于PaddlePaddle实现的模型前向计算网络python代码，以及PaddlePaddle可加载的模型参数文件。  
此外在[[doc](doc/接口速查表.md)]目录中整理了TensorFlow-PaddlePaddle的常用API对比分析。
[环境安装](#环境安装)&nbsp;&nbsp;[使用方法](#使用方法)&nbsp;&nbsp;[验证模型](#验证模型)&nbsp;&nbsp;[注意事项](#注意事项)

## 环境安装

工具开发过程中，我们在如下环境配置中测试模型转换，建议使用[anaconda](https://docs.anaconda.com/anaconda/install)

> python == 2.7 or 3.6

> tensorflow == 1.12.0

> paddlepaddle == 1.3.0

``` shell
# pip install tensorflow-gpu
conda install tensorflow-gpu
pip install paddlepaddle-gpu

# 上述安装过程可能会提示protobuf版本问题
# 升级protobuf解决
pip install protobuf --upgrade
```
         
## 使用方法
本目录下提供了demo示例，展示如何将VGG_16模型转换为PaddlePaddle模型，详见[vgg_translate_tutorial](vgg_translate_tutorial.ipynb)
### 转换模型
```
python tf2fluid/convert.py --pb_file tf_model.pb \
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
**FAQ：输入tensor名和输出tensor名是指什么?**  
TensorFlow模型在infer时，一般调用代码形如`sess.run([output], {input:data})`，其中output即为输出tensor，input则为输入tensor，在进行模型转换时，需提供这input和output对应的`tensor name`，如在[vgg_translate_tutorial](vgg_translate_tutorial.ipynb)中转换VGG_16模型，输入的tensor名为 "inputs", 输出的tensor名为 "vgg_16/fc8/squeezed"

### 转换后模型文件说明  
文件|作用
:------------------:|:-----------------------------------------------:
mymodel.py|基于PaddlePaddle实现的模型网络结构python代码
ref_name.info|my_model.py中各tensor与原TensorFlow模型中的tensor对应关系
const_\*/params_\*|转换后的模型参数文件
save_var.list|模型载入过程中的变量list

## 验证模型
tensorflow2fluid在如下tensorflow模型上测试了模型转换前后的diff  

| 模型类别 | 模型          | Code   | 最大diff |
| -------- | ------------- | ------ | -------- |
| 图像分类 | VGG_16        | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) | 1.04E-05 |
|          | VGG_19        | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) | 9.07E-06 |
|          | ResNet V1 50  | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) | 1.31E-06 |
|          | ResNet V1 101 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) | 4.74E-07 |
|          | Inception V3  | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) | 1.55E-04 |
| 目标检测 | YOLO-Small    | [code](https://github.com/gliese581gg/YOLO_tensorflow) | 1.40E-06 |
|          | YOLO-V3       | [code](https://github.com/mystic123/tensorflow-yolo-v3) | 6.20E-04 |
| 语义分割 | Unet          | [code](https://github.com/jakeret/tf_unet) | 4.17E-07 |

## 注意事项
1. 转换参数`input_format`的设定
> TensorFlow中的CV模型，大多采用`NHWC`的输入格式，但同时也可以支持`NCHW`的格式输入；而在PaddlePaddle中，支持的是`NCHW`的格式。因此需要在转换模型时，指定TensorFlow模型的输入格式，转换过程中会根据输入格式，对输入数据，参数进行变换。

2. 转换参数`input_shape`的设定

> 在模型转换时，需设定输入数据的具体`shape`。因为转换过程中，涉及到较多参数的转换，因此模型转换完成应用到预测时，输入数据的`shape`也须与之前指定的一致，否则可能会出错。

3. 转换参数`use_cuda`的设定

> 受限于PaddlePaddle与TensorFlow部分OP上的实现差异，部分tensor参数（在TensorFlow中，这部分参数类型是tensor类型，但值保持不变）需要通过infer得到。因此模型转换过程中，同时也会加载tensorflow模型进行预测，消耗计算资源。在有GPU资源的的前提下，将`use_cuda`设为`True`有助于提升转换速度。

## Link

本目录下部分代码参考了MMdnn-Tensorflow，对此表示感谢！

[MMdnn-Tensorflow](https://github.com/Microsoft/MMdnn/tree/master/mmdnn/conversion/tensorflow)
