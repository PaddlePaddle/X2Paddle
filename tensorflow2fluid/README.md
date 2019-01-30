# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


tensorflow2fluid支持将训练好的TensorFlow模型转换为PaddlePaddle模型，包括基于PaddlePaddle实现的模型前向计算网络python代码，以及PaddlePaddle可加载的模型参数文件。
> <a href="#环境依赖">`环境依赖`</a>
> <a href="#安装说明">`安装说明`</a>
> <a href="#使用方法">`使用方法`</a>
> <a href="#开发介绍">`开发介绍`</a>
> <a href="#对比实验">`对比实验`</a>

`我们计划专门梳理出指南文档，对比TensorFlow与PaddlePaddle的差异，帮助TensorFlow用户快速上手PaddlePaddle的使用，文档后续会整理在doc目录下，欢迎有需求的同学关注！`

## 环境依赖

> python = 2.7

> tensorflow >= 1.12.0

> 注：tensorflow2fluid的运行不依赖于paddlepaddle，但测试转换后的模型所需的PaddlePaddle须为1.2.0或更新版本

<a id="安装说明">
         
## 安装说明

```
# 如果没有安装paddlepaddle和tensorflow环境
pip install paddlepaddle
pip install tensorflow

git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle/tensorflow2fluid
python setup.py install
```

<a id="使用方法">
         
## 使用方法

> 1. 目前支持转换的模型格式包括checkpoint保存的模型、将参数序列化到网络结构的pb格式模型
> 2. 模型转换后，在输入同样的数据前提下，检查模型转换前后的diff，一般结果最大diff在1e-5数量级

### 转换示例

下面示例中，将vgg_16模型转换至paddlepaddle模型
```
# 下载预训练的vgg_16模型参数
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar xzvf vgg_16_2016_08_28.tar.gz

# 将模型转存为checkpoint格式模型
python demo/export_to_checkpoint.py --model vgg_16 --ckpt_file vgg_16.ckpt --save_dir vgg_checkpoint

# 转换模型
tf2fluid --meta_file vgg_checkpoint/model.meta \
         --ckpt_dir vgg_checkpoint/ \
         --in_nodes inputs \
         --input_shape None,224,224,3 \
         --output_nodes vgg_16/fc8/squeezed \
         --save_dir paddle_vgg
```

### 参数说明

|tf2fluid参数|说明|
|------------------|-----------------------------------------------|
|meta_file|TensorFlow模型序列化后保存的meta文件|
|ckpt_dir|TensorFlow模型保存checkpoint目录|
|pb_file|Tensorflow保存的pb格式模型|
|in_nodes|输入tensor名，多个输入时以空格分隔|
|input_shape|输入tensor的shape(batch维度以None表示)，shape之间以空格分隔，shape内各维度以逗号分隔，须与input_nodes对应|
|output_nodes|输出tensor名，多个输出时以空格分隔|
|save_dir|转换后的模型保存路径|

### 转换后模型文件说明

文件|作用
:------------------:|:-----------------------------------------------:
my_model.py|基于PaddlePaddle实现的模型网络结构python代码
ref_name.txt|my_model.py中各tensor与原TensorFlow模型中的tensor对应关系
const_\*/params_\*|转换后的模型参数文件
save_var.list|模型载入过程中的变量list

### 加载转换后的模型
加载转换后的模型主要注意以下三点

> 1. `import`模型结构，模型结构代码定义在my_model.py中
> 2. 注意原模型中输出与转换后模型输出名的映射关系，参考ref_name.txt
> 3. 模型需要加载的参数列表为save_var.list

仍然以上面转换后的vgg_16为例，下面通过示例展示如何加载模型，并进行预测


**【重要】代码中须注意，PaddlePaddle的图像输入为NCHW格式, 卷积的kernel形状为[filter_num, in_channel, height, width]， 卷积输出形状为[batch, filter_num, height, width]，这三点与tensorflow默认情况均不同**

```
#coding:utf-8
# paddle_vgg为转换后模型存储路径
from paddle_vgg.mymodel import KitModel
import paddle.fluid as fluid
import numpy

def model_initialize():
    # 构建模型结构，并初始化参数
    result = KitModel()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # 根据save_var.list列表，加载模型参数
    var_list = list()
    global_block = fluid.default_main_program().global_block()
    with open('paddle_vgg/save_var.list') as f:
        for line in f:
            try:
                # 过滤部分不需要加载的参数（OP配置参数）
                var = global_block.var(line.strip())
                var_list.append(var)
            except:
                pass
    fluid.io.load_vars(exe, 'paddle_vgg', vars=var_list)

    prog = fluid.default_main_program()
    return exe, prog, result
    
def test_case(exe, prog, result):
    # 测试随机数据输入
    numpy.random.seed(13)
    img_data = numpy.random.rand(1, 224, 224, 3)
    # tf中输入为NHWC，PaddlePaddle则为NCHW，需transpose
    img_data = numpy.transpose(img_data, (0, 3, 1, 2))

    # input_0为输入数据的张量名，张量名和数据类型须与my_model.py中定义一致
    r, = exe.run(fluid.default_main_program(),
                feed={'input_0':numpy.array(img_data, dtype='float32')},
                fetch_list=[result])
    
    # 调用save_inference_model可将模型结构（当前以代码形式保存）和参数均序列化保存
    # 保存后的模型可使用load_inference_model加载
    # http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/api_guides/low_level/inference.html#api-guide-inference
    fluid.io.save_inference_model("./paddle_model", ["input_0"], [result], exe)
    
if __name__ == "__main__":
    exe, prog, result = model_initialize()
    test_case(exe, prog, result)
```

<a id="开发介绍">
         
## 开发介绍

tensorflow2fluid在模型转换过程中，以tensorflow计算图中的节点为粒度，遍历图中的节点，并将每个节点所对应的OP转换为基于PaddlePaddle实现的python网络结构代码。

> 模型中所使用的代码，一般而言并不能直接能过模型训练时所使用的tensorflow代码中就能完全看出来。比如在python模型代码中所使用到的`tf.contrib.layers.fully_connected`就涉及到如下OP

|TensorFlow OP名|说明|
|:-----------------:|:----------------------------------------:|
|VariableV2|用于创建变量weights和bias|
|MatMul|输入与weights乘法操作|
|BiasAdd|输入值在Matmul后，再与bias相加|
|Relu|输出最后需要通过的激活函数操作|
|Idenitity|计算过程中的变量复制操作|

目前支持转换OP如文档最末附表所示，需要注意的是，**在实现转换过程中，代码转换基于各OP常见的使用情况**，此外，并非所有OP都需要转成PaddlePaddle对应的代码实现，如Identity，switch等OP，在实际转换过程中，都直接将输出表示为输入即可。

| TensorFlow OP       | Python Api | TensorFlow OP          | Python Api |
| ------------------- | ---------- | ---------------------- | ---------- |
| VariableV2          | 1          | placeholderwithdefault | 17         |
| Identity            | 2          | switch                 | 18         |
| Placeholder         | 3          | merge                  | 19         |
| Const               | 4          | MaxPool                | 20         |
| Conv2D              | 5          | Squeeze                | 21         |
| BiasAdd             | 6          | Add                    | 22         |
| Relu                | 7          | Mean                   | 23         |
| Conv2dBackpropInput | 8          | DepthwiseConv2dNative  | 24         |
| FusedBatchNorm      | 9          | Pad                    | 25         |
| ConcatV2            | 10         | StridedSlice           | 26         |
| AvgPool             | 11         | ResizeNearestNeighbor  | 27         |
| Rsqrt               | 12         | Maximum                | 28         |
| Mul                 | 13         | Minimum                | 9          |
| Sub                 | 14         | Sigmoid                | 30         |
| Shape               | 15         | Pack                   | 31         |
| Reshape             | 16         |                        |            |

tensorflow2paddle仍在持续开发阶段中，也非常欢迎用户贡献自己的代码，或者通过issue的方式提出建议和需求。

<a id="对比实验">
         
## 对比实验

tensflow2fluid在公开的TensorFlow预训练模型上，通过输入1000个随机数据在原模型和转换后的模型上进行预测，得到的平均diff大小如下表所示

Model|Pre-trained Model|Average Diff|Max Diff
:--------------:|:----------------------------------------------:|:-----------------:|:-----------------:
[vgg_16](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py)|[vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)|-|-
[vgg_19](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py)|[vgg_19_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)|-|-
[resnet_v1_50](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py)|[resnet_v1_50_2016_08_28.tar.gz](http://download.tenlssorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)|-|-
[resnet_v1_101](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py)|[resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)|-|-
[inception_v3](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py)|[inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)|-|-

## Link

[MMdnn-Tensorflow](https://github.com/Microsoft/MMdnn/tree/master/mmdnn/conversion/tensorflow)
