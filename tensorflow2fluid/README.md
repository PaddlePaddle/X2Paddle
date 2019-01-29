# tensorflow2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

## 环境依赖

> python = 2.7

> tensorflow >= 1.12.0

> 注：tensorflow2fluid的运行不依赖于paddlepaddle，但测试转换后的模型所需的PaddlePaddle须为1.2.0或更新版本

## 安装说明

```
# 如果没有安装paddlepaddle和tensorflow环境
pip install paddlepaddle
pip install tensorflow

git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle/tensorflow2fluid
python setup.py install
```

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
