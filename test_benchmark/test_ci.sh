#docker 命令参考
#nvidia-docker run -it --cpu-shares=20000 --name=x2paddle --rm -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi -v $(pwd):/workspace paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82 /bin/bash

#下载最新dev paddle/torch/torchvison
wget -nc https://paddle-qa.bj.bcebos.com/PaddleX2paddle/torch-1.7.0-cp37-cp37m-linux_x86_64.whl 
wget -nc https://paddle-qa.bj.bcebos.com/PaddleX2paddle/torchvision-0.8.1-cp37-cp37m-linux_x86_64.whl 
wget -nc https://paddle-wheel.bj.bcebos.com/2.4.2/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.4.2.post112-cp37-cp37m-linux_x86_64.whl

#设定python版本
python --version
#which python
ls /usr/bin
unlink /usr/bin/python
ln -s /usr/bin/python3.7 /usr/bin/python
python --version

#python -m pip config set global.index-url https://mirror.baidu.com/pypi/simple;
python -m pip install --upgrade pip;
python -m pip install wget;
python -m pip install timm;
python -m pip install transformers==3.1.0;
python -m pip install ./*.whl;
python -m pip install pandas;
python -m pip install nose;
python -m pip install pytest;
python -m pip install opencv-python==4.6.0.66;
python -m pip install allure-pytest;
python -m pip install pynvml psutil GPUtil;
python -m pip install sympy;
python -m pip install treelib;
python -m pip install tensorflow==1.14.0;
python -m pip install onnx==1.8.0;
python -m pip install easyocr==1.2.1;
python -m pip install torchmetrics==0.10.2;
python -m pip install pytorch_lightning==1.5.3;
python -m pip install kornia==0.5.11;

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH};
# build x2paddle
cd ..;
python setup.py install;
 if [ "$?" != "0" ]; then
    echo -e "\033[33m build x2-paddle failed! \033[0m";
    exit -1;
fi;

cd test_benchmark/Caffe;
bash convert.sh;

cd ../PyTorch;
mv YOLOX ..; #暂时取消
bash convert.sh;

cd ../ONNX;
bash convert.sh;

cd ../TensorFlow;
mv KerasBert ..; #暂时取消，会hang
bash convert.sh;
