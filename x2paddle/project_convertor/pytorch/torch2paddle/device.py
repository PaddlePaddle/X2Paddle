import os
import paddle

def device_count():
    gpu_useful = paddle.get_device().startswith("gpu")
    if gpu_useful:
        device_str = os.environ["CUDA_VISIBLE_DEVICES"]
        seg = device_str.split(",")
        return len(seg)
    else:
        return 0
    