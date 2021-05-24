## torch.nn.ReplicationPad3d
### [torch.nn.ReplicationPad3d](https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad3d.html?highlight=pad#torch.nn.ReplicationPad3d)
```python
torch.nn.ReplicationPad3d(padding)
```
### [paddle.nn.Pad3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Pad3D_cn.html#pad3d)
```python
paddle.nn.Pad3D(padding, mode='constant', value=0.0, data_format='NCDHW', name=None)
```

### 功能差异

#### 使用方式
***PyTorch***：只支持`replicate`方式的Pad方式。  
***PaddlePaddle***：支持`constant`、`reflect`、`replicate`、`circular`四种格式的输入（通过`mode`设置）。

#### 输入格式
***PyTorch***：只支持`NCDHW`的输入。  
***PaddlePaddle***：支持`NCDHW`和`NCDHW`两种格式的输入（通过`data_format`设置）。

#### padding的设置
***PyTorch***：padding参数的类型只能为int或tuple。  
***PaddlePaddle***：padding参数的类型只能为Tensor或list。


### 代码示例
``` python
# PyTorch示例：
import paddle
import paddle.nn as nn
import numpy as np
input_shape = (1, 1, 1, 2, 3)
pad = [1, 0, 1, 2, 0, 0]
mode = "constant"
data = paddle.arange(np.prod(input_shape), dtype="float32").reshape(input_shape) + 1
my_pad = nn.Pad3D(padding=pad, mode=mode)
result = my_pad(data)
# 输出
# tensor([[[[[1., 1., 2., 3.],
#            [1., 1., 2., 3.],
#            [4., 4., 5., 6.],
#            [4., 4., 5., 6.],
#            [4., 4., 5., 6.]]]]])

```

``` python
# PaddlePaddle示例：
import paddle
import paddle.nn as nn
import numpy as np
input_shape = (1, 1, 1, 2, 3)
pad = [1, 0, 1, 2, 0, 0]
mode = "constant"
data = paddle.arange(np.prod(input_shape), dtype="float32").reshape(input_shape) + 1
my_pad = nn.Pad3D(padding=pad, mode=mode)
result = my_pad(data)
# 输出
# Tensor(shape=[1, 1, 1, 5, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[[[1., 1., 2., 3.],
#            [1., 1., 2., 3.],
#            [4., 4., 5., 6.],
#            [4., 4., 5., 6.],
#            [4., 4., 5., 6.]]]]])
```
