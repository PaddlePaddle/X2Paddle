# Converting PyTorch's Yolov5 to PaddlePaddle

We can use [ultralytics/yolov5](https://github.com/ultralytics/yolov5) and [model files](https://pytorch.org/hub/ultralytics_yolov5/) for model conversion.

## Model conversion

The following is sample code:


``` python


import torch
import numpy as np

# 1. load model
torch_model = torch.hub.load(
    '../dataset/yolov5/yolov5',
    source='local',
    model='custom',
    path='../dataset/yolov5/yolov5s.pt')
torch_model.eval()

# 2. load input img
img = np.load("../dataset/yolov5/input.npy")

# 3. trace once
# https://github.com/ultralytics/yolov5/issues/9341
# torch need to load yolo model twice!!!
# so, we load one time before converting to paddle!!!
try:
    torch.jit.trace(torch_model, torch.tensor(img))
except:
    pass


# 4. convert model
from x2paddle.convert import pytorch2paddle

save_dir = "pd_model_trace"
jit_type = "trace"

pytorch2paddle(torch_model,
               save_dir,
               jit_type,
               [torch.tensor(img)],
               disable_feedback=True)

```

It can be roughly divided into four steps:

1. Load the model

The model can be loaded using the method provided by [YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/) or locally.

Here the interface `torch.hub.load` is used, where the meaning of each parameter is:

- `'... /dataset/yolov5/yolov5'`, which indicates the directory where the repo for `ultralytics/yolov5` is located. Therefore, you need to manually pull [ultralytics/yolov5](https://github.com/ultralytics/yolov5) into the appropriate directory first.
- `source='local'` means to use a local repo, which corresponds to the first parameter above.
- `model='custom'`, means use local model, download the model in [model files](https://pytorch.org/hub/ultralytics_yolov5/) to the corresponding directory.
- `path='... /dataset/yolov5/yolov5s.pt'`, indicates the location of the local model files, which corresponds to third parameter above.

2. Loading test data

The data type loaded here is `numpy`. Note that the model outputs different types of results for different data types.

3. First trace

According to the discussion in https://github.com/ultralytics/yolov5/issues/9341, the yolov5 model may report an error when using `torch.jit.trace` and needs to be traced twice!

Therefore, it is important to trace once here, otherwise subsequent conversions will fail.

4. Converting Models

The interface of conversion model can be called directly.

Because of the problems mentioned in (3), it is recommended to use the interface call to convert the model, as the CLI command mode conversion may cause errors.

Yolov5 models can not be converted using `script`, please check with PyTorch and [ultralytics/yolov5](https://github.com/ultralytics/yolov5) to see if they have followed up on this issue.

## Model usage

After successful conversion, `pd_model_trace` will be generated in the current directory, where the related model structure code and model files are stored and can be used directly.For example:

``` python

import paddle
import numpy as np

img = np.load("../dataset/yolov5/input.npy")

# trace
paddle.enable_static()
exe = paddle.static.Executor(paddle.CPUPlace())
[prog, inputs, outputs] = paddle.static.load_inference_model(
    path_prefix="pd_model_trace/inference_model/model", executor=exe)
result = exe.run(prog, feed={inputs[0]: img}, fetch_list=outputs)

```
