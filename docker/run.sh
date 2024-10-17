#生成镜像
docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t paddlepaddle/x2paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82 .

#进入镜像
nvidia-docker run -it --cpu-shares=20000 --name=user-x2paddle --rm -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi -v $(pwd):/workspace paddlepaddle/x2paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82 /bin/bash
