# Environment Installation
The caffe2fluid is tested in the following environment configuration. In order to meet the environment dependence of the caffe2fluid, users can configure their own environment according to the following process, or configure according to their own needs.

## 1. Anaconda Installation
Directly refer to the official website installation documentation.  
[Install in Linux](https://docs.anaconda.com/anaconda/install/linux/)    
[Install in Mac](https://docs.anaconda.com/anaconda/install/mac-os/)  

## 2.Create Python Environment
Create a python environment by using anaconda. Then install Caffe and PaddlePaddle in the created python environment. The created environment can be independent of the system environment, so the modifications to the creation environment will not affect the dependencies of other environments or systems.  
```shell
# Create the environment which is named as caffe_paddle, 
# and the version of python is 3.5.
conda create -n caffe-paddle python=3.5

# Activate the environment.
source activate caffe-paddle

# Install the PaddlePaddle and Caffe.
# After installion，run "import caffe" and "import paddle.fluid"
# to determine if it has been installed successfully.
pip install paddlepaddle-gpu
conda install caffe-gpu

# Install the future module of python。
pip install future


# Note: Due to the protobuf version, the installation framework should first install PaddlePaddle and then install Caffe.
# If you installed Caffe first, after installing PaddlePaddle you can solve by the following steps.
pip uninstall protobuf
pip install protobuf==3.6.0

source deactivate
```
