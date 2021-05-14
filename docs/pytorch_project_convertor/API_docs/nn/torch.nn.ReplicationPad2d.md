## torch.nn.ReplicationPad2d
### [torch.nn.ReplicationPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html?highlight=pad#torch.nn.ReplicationPad2d)
```python
torch.nn.ReplicationPad2d(padding)
```
### [paddle.nn.Pad2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/common/Pad2D_cn.html#pad2d)
```python
paddle.nn.Pad2D(padding, mode='constant', value=0.0, data_format='NCHW', name=None)
```

### 功能差异

#### 使用方式
***PyTorch***：只支持`replicate`方式的Pad方式。  
***PaddlePaddle***：支持`constant`、`reflect`、`replicate`、`circular`四种格式的输入（通过`mode`设置）。

#### 输入格式
***PyTorch***：只支持`NCHW`的输入。  
***PaddlePaddle***：支持`NCHW`和`NHWC`两种格式的输入（通过`data_format`设置）。

#### padding的设置
***PyTorch***：padding参数的类型只能为int或tuple。  
***PaddlePaddle***：padding参数的类型只能为Tensor或list。


### 代码示例
``` python
# PyTorch示例：
import torch
import torch.nn as nn
import numpy as np
input_shape = (1, 1, 2, 3)
pad = [1, 0, 1, 0]
data = torch.arange(np.prod(input_shape), dtype=torch.float32).reshape(input_shape) + 1
my_pad = nn.ReplicationPad2d(padding=pad)
result = my_pad(data)
# 输出
# tensor([[[[1., 1., 2., 3.],
#           [1., 1., 2., 3.],
#           [4., 4., 5., 6.]]]])
```

``` python
# PaddlePaddle示例：
import paddle
import paddle.nn as nn
import numpy as np
input_shape = (1, 1, 2, 3)
pad = [1, 0, 1, 0]
mode = "replicate"
data = paddle.arange(np.prod(input_shape), dtype="float32").reshape(input_shape) + 1
my_pad = nn.Pad2D(padding=pad, mode=mode)
result = my_pad(data)
# 输出
# Tensor(shape=[1, 1, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[[1., 1., 2., 3.],
#           [1., 1., 2., 3.],
#           [4., 4., 5., 6.]]]])
```
