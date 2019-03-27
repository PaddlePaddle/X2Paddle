# 环境安装
caffe2fluid在如下环境配置中进行测试，用户可按如下流程配置自己的环境，也可根据自己需求配置，满足caffe2fluid运行对环境的依赖即可。

## 1. 安装Anaconda
可直接参考官网安装文档
[Linux下安装](https://docs.anaconda.com/anaconda/install/linux/)  
[Mac下安装](https://docs.anaconda.com/anaconda/install/mac-os/)

## 2.创建Python环境
通过使用anaconda，创建python环境，在创建的python环境中安装Caffe和PaddlePaddle，创建的环境可以独立于系统环境，对创建环境的修改，也不会影响其它环境或系统的依赖。
```shell
# 创建名为caffe_paddle的环境，python版本指定为3.5
conda create -n caffe-paddle python=3.5

# 激活环境
source activate caffe-paddle

# 安装PaddlePaddle和caffe
# 安装后，可在python中执行"import caffe"和
# "import paddle.fluid"，判断是否已经安装成功
pip install paddlepaddle-gpu
conda install caffe-gpu

# 安装Python的future模块
pip install future
# 退出环境
source deactivate
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
