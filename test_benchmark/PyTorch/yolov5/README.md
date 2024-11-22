# 转换 PyTorch 的 Yolov5 至 PaddlePaddle 模型

我们可以使用 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 和 [model files](https://pytorch.org/hub/ultralytics_yolov5/) 进行模型的转换。

## 模型转换

以下为示例代码：


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

大体可分为四个步骤：

1. 加载模型

可以使用 [YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/) 提供的加载模型方式，也可以采用上述代码中的本地加载方式。

这里使用接口 `torch.hub.load`，其中，各个参数的意义为：

- `'../dataset/yolov5/yolov5'`，表示 `ultralytics/yolov5` 的 repo 所在的目录。因此，需要先手动把 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 拉取到相应的目录中。
- `source='local'`，表示使用本地 repo，对应上述第一个参数。
- `model='custom'`，表示此处使用本地的模型，也可以把 [model files](https://pytorch.org/hub/ultralytics_yolov5/) 中的模型下载到相应目录。
- `path='../dataset/yolov5/yolov5s.pt'`，表示本地模型文件的位置，对应上述第三个参数。

2. 加载测试数据

此处加载的数据类型为 `numpy`。注意，不同的数据类型，模型的输出结果类型不同。

3. 第一次 trace

根据 https://github.com/ultralytics/yolov5/issues/9341 中的讨论，yolov5 模型在使用 `torch.jit.trace` 时可能会报错，而且需要 trace 两次！

因此，此处务必先 trace 一次，否则后续转换出错。

4. 转换模型

直接调用接口转换模型即可。

由于 (3) 中提到的问题，因此，建议采用接口调用的方式进行模型转换，CLI 的命令模式转换方式可能出错。

另外，yolov5 的模型使用 `script` 方式转换也可能出错，请关注 PyTorch 与 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 是否有跟进此问题。

## 模型使用

转换成功后，会在当前目录生成 `pd_model_trace` ，相关模型结构代码与模型文件存储在此处，可直接使用。如：

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
