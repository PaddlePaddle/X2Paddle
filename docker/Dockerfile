FROM paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82

ENV HTTP_PROXY <HTTP_PROXY>
ENV HTTPS_PROXY <HTTPS_PROXY>
RUN export http_proxy=$HTTP_PROXY
RUN export https_proxy=$HTTPS_PROXY

RUN python -m pip install --upgrade pip && \
    python -m pip install wget timm transformers pandas nose pytest opencv-python==4.6.0.66 allure-pytest && \
    python -m pip install torch==2.4.1 torchvision torchaudio tensorflow==2.16.1 onnx==1.17.0 onnxruntime && \
    python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
    python -m pip install pynvml psutil GPUtil sympy treelib easyocr && \
    python -m pip install torchmetrics pytorch_lightning kornia==0.5.11 hypothesis pre-commit==2.17.0 protobuf==3.20.2 && \
    python -m pip install --upgrade virtualenv
