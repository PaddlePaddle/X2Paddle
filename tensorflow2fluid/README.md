# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

## 依赖

> python = 2.7

> tensorflow >= 1.12.0

tensorflow2fluid的模型转换不依赖paddlepaddle

## 介绍

tensorflow2fluid支持将训练好的TensorFlow模型转至PaddlePaddle模型，转换后的保存目录中，文件list如下表所示

|文件|作用|
|------------------|-----------------------------------------------|
|my_model.py|基于PaddlePaddle实现的模型网络结构python代码|
|ref_name.txt|my_model.py中各tensor与原TensorFlow模型中的tensor对应关系|
|const_*/params_*|转换后的模型参数文件|

## 用法
使用tensorflow2fluid转换模型时，所需的信息如下

|参数|说明|
|------------------|-----------------------------------------------|
|meta_file|TensorFlow模型序列化后保存的meta文件|
|ckpt_file|TensorFlow保存的ckpt格式模型参数|
|pb_file|Tensorflow保存的pb格式模型|
|input_nodes|输入tensor名，多个输入时以空格分隔|
|input_shape|输入tensor的shape(batch维度以None表示)，shape之间以空格分隔，shape内各维度以逗号分隔，须与input_nodes对应|
|output_shape|输出tensor名，多个输出时以空格分隔|
|save_dir|转换后的模型保存路径|

目前TensorFlow保存的模型主要包括ckpt和pb两种类型。其中加载ckpt模型时，同时也需通过meta文件导入网络结构；而pb模型则已将网络结构和参数均序列化至同一个文件。因此，加载ckpt模型时，需指定meta_file和ckpt_file，而加载pb模型，则只需指定pb_file即可。
