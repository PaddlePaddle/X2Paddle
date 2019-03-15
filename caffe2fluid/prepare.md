# 环境安装
## 1.[安装Anaconda（linux python=3.7 anaconda=3.）](https://docs.anaconda.com/anaconda/install/)
## 2.创建Python环境

在使用Caffe2Fluid这一工具时需要同时使用Caffe和PaddlePaddle，我们需要创建一个环境在安装了Caffe和PaddlePaddle的同时，不出现一俩影响问题。
> ```shell
> # 创建名为caffe_paddle的环境，python版本指定为3.5
> conda create -n caffe-paddle python=3.5
>
> # 环境创建完后，激活环境
> source activate caffe-paddle
> # 安装PaddlePaddle
> pip install paddlepaddle-gpu
> # 安装Caffe
> conda install caffe-gpu
> # 测试是否已经完成Caffe和PaddlePaddle，直接在终端输入python命令，同时输入"import caffe"和"import paddle.fluid"，若无出错则表示已将安装成功
> # 退出环境
> source deactivate
> # 为了更便利地使用这一环境，需要将环境名与环境路径相匹配
> vim ~/.bashrc
> alias caffe-paddle=“Anaconda安装路径/envs/caffe-paddle/bin/python"
>source ~/.bashrc
> ```
注意：由于protobuf版本问题，安装框架过程应先安装PaddlePaddle，再安装Caffe。如若先安装了Caffe，则可以在安装PaddlePaddle后执行下述命令行。
> ```shell
> pip uninstall protobuf
> pip install protobuf==3.6.0
> ```
