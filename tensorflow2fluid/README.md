# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


tensorflow2fluid支持将训练好的TensorFlow模型转换为PaddlePaddle模型，包括基于PaddlePaddle实现的模型前向计算网络python代码，以及PaddlePaddle可加载的模型参数文件。
> <a href="#环境依赖">`环境依赖`</a>
> <a href="#使用方法">`使用方法`</a>
> <a href="#开发介绍">`开发介绍`</a>
> <a href="#对比实验">`对比实验`</a>

**我们计划专门梳理出指南文档，对比TensorFlow与PaddlePaddle的差异，帮助TensorFlow用户快速上手PaddlePaddle的使用，文档后续会整理在doc目录下，欢迎有需求的同学关注！**

## 依赖环境

工具开发过程中，我们在如下环境配置中测试模型转换

> python == 2.7 or 3.6

> tensorflow == 1.12.0

> paddlepaddle == 1.3.0

<a id="使用方法">
         
## 使用方法
```
python src/convert.py --pb_file tf_model.pb \
                      --in_nodes inputs \
                      --output_nodes outputs \
                      --input_shape None,224,224,3 \
                      --input_format NHWC \
                      --save_dir paddle_model
```
### 参数说明  
|tf2fluid参数|说明|
|-----------|-----------------------------------------------|
|meta_file|TensorFlow模型序列化后保存的meta文件|
|ckpt_dir|TensorFlow模型保存checkpoint目录|
|pb_file|Tensorflow保存的pb格式模型|
|in_nodes|输入tensor名，多个输入时以空格分隔|
|input_shape|输入tensor的shape(batch维度以None表示)，shape之间以空格分隔，shape内各维度以逗号分隔，须与input_nodes对应|
|input_format|输入数据格式，当为CV模型时，可选择NHWC/NCHW/OTHER|
|output_nodes|输出tensor名，多个输出时以空格分隔|
|save_dir|转换后的模型保存路径|

目前支持tensorflow保存的checkpoint模型和将参数及模型结构序列化存储的pb模型，前者须指定meta_file和ckpt_file，后者则指定pb_file

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

> 1. 目前支持转换的模型格式包括checkpoint保存的模型、将参数序列化到网络结构的pb格式模型
> 2. 模型转换后，计算结果存在一定精度的diff，因此务必检查模型转换前后，在输入同样的数据前提下,diff是否符合预期


## Link

本目录下部分代码参考了MMdnn-Tensorflow，对此表示感谢！

[MMdnn-Tensorflow](https://github.com/Microsoft/MMdnn/tree/master/mmdnn/conversion/tensorflow)
