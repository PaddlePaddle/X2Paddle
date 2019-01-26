# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

## 依赖

> python = 2.7

> tensorflow >= 1.12.0

> 注：tensorflow2fluid的运行不依赖于paddlepaddle，但测试转换后的模型所需的PaddlePaddle须为1.2.0或更新版本

## 介绍

tensorflow2fluid支持将训练好的TensorFlow模型转至PaddlePaddle fluid模型，转换后的保存目录中，文件list如下表所示

文件|作用
:------------------:|:-----------------------------------------------:
my_model.py|基于PaddlePaddle实现的模型网络结构python代码
ref_name.txt|my_model.py中各tensor与原TensorFlow模型中的tensor对应关系
const_\*/params_\*|转换后的模型参数文件

tensorflow2fluid在模型转换过程中，以tensorflow计算图中的节点为粒度，遍历图中的节点，并将每个节点所对应的OP转换为基于PaddlePaddle实现的python网络结构代码，目前支持OP如下表所示

TensorFlow OP名|Python接口
:---------------------------:|:------------------------------------------:|
conv2d|tf.layers.conv2d

模型中所使用的代码，一般而言并不能直接能过模型训练时所使用的tensorflow代码中就能完全看出来。比如在python模型代码中所使用到的`tf.contrib.layers.fully_connected`就涉及到如下OP

TensorFlow OP名|说明
:-----------------:|:----------------------------------------:
VariableV2|用于创建变量weights和bias
MatMul|输入与weights乘法操作
BiasAdd|输入值在Matmul后，再与bias相加
Relu|输出最后需要通过的激活函数操作
Idenitity|计算过程中的变量复制操作

## 模型转换diff对比

tensflow2fluid在公开的TensorFlow预训练模型上，通过输入随机数据在原模型和转换后的模型上进行预测，得到的平均diff大小如下表所示

Model|Pre-trained Model|Diff
:--------------:|:----------------------------------------------:|:-----------------:
[vgg_16](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py)|[inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)|1e-05
[vgg_19](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py)|[vgg_19_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)|1e-05
[resnet_v1_50](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py)|[resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)|1e-05
[resnet_v1_101](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py)|[resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)|1e-05
[inception_v3](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py)|[inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)|1e-05

## 用法
使用tensorflow2fluid转换模型时，所需的信息如下

|参数|说明|
|------------------|-----------------------------------------------|
|meta_file|TensorFlow模型序列化后保存的meta文件|
|ckpt_file|TensorFlow模型保存checkpoint目录|
|pb_file|Tensorflow保存的pb格式模型|
|input_nodes|输入tensor名，多个输入时以空格分隔|
|input_shape|输入tensor的shape(batch维度以None表示)，shape之间以空格分隔，shape内各维度以逗号分隔，须与input_nodes对应|
|output_shape|输出tensor名，多个输出时以空格分隔|
|save_dir|转换后的模型保存路径|

目前TensorFlow保存的模型主要包括ckpt和pb两种类型。其中加载ckpt模型时，同时也需通过meta文件导入网络结构；而pb模型则已将网络结构和参数均序列化至同一个文件。因此，加载ckpt模型时，需指定meta_file和ckpt_file，而加载pb模型，则只需指定pb_file即可。

### 例：将inception_v3模型转换至PaddlePaddle

```Bash
# 下载并解压inception_v3预训练模型
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar xzvf http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

# 将下载的预训练模型转存为check_point
python demo/inception_v3/export_to_checkpoint.py inception_v3.py checkpoint

# 将check_point模型转换为PaddlePaddle可加载运行的模型
python convert.py --meta_file checkpoint/model.meta \
                  --ckpt_dir checkpoint \
                  --in_nodes inputs \
                  --input_shape None,299,299,3 \
                  --output_nodes InceptionV3/Logits/SpatialSqueeze \
                  --save_dir paddle_inception_v3
```

### 加载转换后的模型

## Link
[MMdnn-Tensorflow](https://github.com/Microsoft/MMdnn/tree/master/mmdnn/conversion/tensorflow)
