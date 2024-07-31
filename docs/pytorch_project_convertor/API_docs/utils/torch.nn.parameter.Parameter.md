## torch.nn.parameter.Parameter
### [torch.nn.parameter.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html?highlight=torch%20nn%20parameter#torch.nn.parameter.Parameter)
```python
torch.nn.parameter.Parameter(data, requires_grad=True)
```

## [paddle.create_parameter](https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/fluid/layers/tensor.py#L77)
```python
paddle.create_parameter(shape,
                       dtype,
                       name=None,
                       attr=None,
                       is_bias=False,
                       default_initializer=None)
```



### 功能差异

#### 使用方式
***PyTorch***：通过设置`data`将Tensor赋给Parameter。
***PaddlePaddle***：有2种方式创建Parameter。方式一：通过设置`attr`将ParamAttr赋给Parameter；方式二：通过设置`shape`（大小）、`dtype`（类型）、`default_initializer`（初始化方式）设置Parameter。

#### 梯度设置
***PyTorch***：通过设置`requires_grad`确定是否进行梯度反传。
***PaddlePaddle***：PaddlePaddle无此功能。



### 代码示例
``` python
# PyTorch示例：
import torch
x = torch.zeros(2, 3)
param = torch.nn.parameter.Parameter(x, requires_grad=False)

# 输出
# Parameter containing:
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
```

``` python
# PaddlePaddle示例：
import paddle
x = paddle.zeros([2, 3], dtype="float32")
param = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
param.stop_gradient = True

# 输出
# Parameter containing:
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[0., 0., 0.],
#         [0., 0., 0.]])
```
