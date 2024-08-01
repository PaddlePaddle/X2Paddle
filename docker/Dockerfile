FROM paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82

RUN wget https://paddle-qa.bj.bcebos.com/PaddleX2paddle/torch-1.7.0-cp37-cp37m-linux_x86_64.whl --no-proxy && \
    wget https://paddle-qa.bj.bcebos.com/PaddleX2paddle/torchvision-0.8.1-cp37-cp37m-linux_x86_64.whl --no-proxy && \
    wget https://paddle-wheel.bj.bcebos.com/2.4.2/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.4.2.post112-cp37-cp37m-linux_x86_64.whl --no-proxy

RUN python --version && unlink /usr/bin/python && ln -s /usr/bin/python3.7 /usr/bin/python

ENV HTTP_PROXY <HTTP_PROXY>
ENV HTTPS_PROXY <HTTPS_PROXY>
RUN export http_proxy=$HTTP_PROXY
RUN export https_proxy=$HTTPS_PROXY

RUN python -m pip install --upgrade pip && \
    python -m pip install wget timm transformers==3.1.0 ./*.whl pandas nose pytest opencv-python==4.6.0.66 allure-pytest && \
    python -m pip install pynvml psutil GPUtil sympy treelib tensorflow==1.14.0 onnx==1.10.0 onnxruntime==1.10.0 easyocr==1.2.1 && \
    python -m pip install torchmetrics==0.10.2 pytorch_lightning==1.5.3 kornia==0.5.11 hypothesis pre-commit==2.17.0 && \
    python -m pip install --upgrade virtualenv
