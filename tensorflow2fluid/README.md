# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

## Dependency

> python = 2.7

> paddlepaddle >= 1.2.0

> tensorflow >= 1.12.0

## Introduce

tensorflow2fluid支持将训练好的TensorFlow模型转至PaddlePaddle模型，转换后的保存目录中，文件list如下表所示

|文件|作用|
|------------------|-----------------------------------------------|
|my_model.py|基于PaddlePaddle实现的模型网络结构python代码|
|ref_name.txt|my_model.py中各tensor与原TensorFlow模型中的tensor对应关系|
|const_*/params_*|转换后的模型参数文件|

## Usage
使用tensorflow2fluid转换模型时，所需的信息如下

|参数|说明|
|------------------|-----------------------------------------------|
|input_nodes|输入tensor名，多个输入时以空格分隔|
|input_shape|输入tensor的shape(batch维度以None表示)，shape之间以空格分隔，shape内各维度以逗号分隔，须与input_nodes对应|
|output_shape|输出tensor名，多个输出时以空格分隔|
|save_dir|转换后的模型保存路径|
