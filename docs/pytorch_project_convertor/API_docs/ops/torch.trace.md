## torch.trace
### [torch.trace](https://pytorch.org/docs/stable/generated/torch.trace.html?highlight=trace#torch.trace)
```python
torch.trace(input) 
```
### [paddle.trace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/trace_cn.html)
```python
paddle.trace(x, offset=0, axis1=0, axis2=1, name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x            | 表示输入的Tensor。               |
| -        | offset            | 2D-Tensor中获取对角线的位置，默认值为0，即主对角线，PyTorch无此参数。                  |
| -           |axis1            | 当输入的Tensor维度大于2D时，获取对角线的二维平面的第一维，PyTorch无此参数。               |
| -        | axis2            | 当输入的Tensor维度大于2D时，获取对角线的二维平面的第二维，PyTorch无此参数。                   |

### 代码示例
``` python
# PyTorch示例：
import torch
x = torch.arange(1., 10.).view(3, 3)
# 输入：
# tensor([[1., 2., 3.],
#         [4., 5., 6.],
#         [7., 8., 9.]])
out = torch.trace(x)
# 输出：
# tensor(15.)
```

``` python
# PaddlePaddle示例：
import paddle
x = paddle.arange(1., 10.).reshape([3, 3])
# 输入：
# Tensor(shape=[3, 3], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])
out0 = paddle.trace(x)
# 输出out0：
# Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [15])
out1 = paddle.trace(x, offset=1)
# 输出out1：
# Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [8])
```