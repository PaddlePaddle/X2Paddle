# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

## Dependency

> python = 2.7

> paddlepaddle >= 1.2.0

> tensorflow >= 1.12.0

## Introduce

tensorflow2fluid支持将训练好的TensorFlow模型转至PaddlePaddle模型，转换后的文件list如下

> save_dir
>    |
>    |----my_model.py    基于PaddlePaddle实现的模型网络结构python代码
>    |----ref_name.txt    my_model.py中各输出与原TensorFlow模型中的输出对应关系
>    |----const_*/params_*    转换后的模型参数文件
