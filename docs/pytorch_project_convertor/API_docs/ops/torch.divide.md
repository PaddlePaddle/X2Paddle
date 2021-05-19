## torch.divide
### [torch.divide](https://pytorch.org/docs/stable/generated/torch.divide.html?highlight=divide#torch.divide)
```python
torch.divide(input, other, *, rounding_mode=None, out=None)
```

### [paddle.divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/math/divide_cn.html#divide)
```python
paddle.divide(x, y, name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| rounding_mode | -        | 表示舍入模式，PaddlePaddle无此参数。  |
| out          | -        | 表示输出的Tensor，PaddlePaddle无此参数。  |

### 功能差异

#### 舍入模式设置
***PyTorch***：可以通过`rounding_mode`设置舍入模式，`"trunc"`表示向0取整，`"floor"`表示向下取整，默认值为`None`表示不进行任何舍入操作。  
***PaddlePaddle***：PaddlePaddle无此功能，需要组合实现。


### 代码示例
``` python
# PyTorch示例：
import torch
a = torch.tensor([ 0.3810,  1.2774, -0.3719,  0.4637])
b = torch.tensor([ 1.8032,  0.2930, 0.5091, -0.1392])
out = torch.divide(a, b, rounding_mode='trunc')
# 输出
# tensor([ 0.,  4., -0., -3.])
```

``` python
# PaddlePaddle示例：
import paddle
a = paddle.to_tensor([0.3810,  1.2774, -0.3719, 0.4637], dtype="float32")
b = paddle.to_tensor([1.8032,  0.2930, 0.5091, -0.1392], dtype="float32")
ipt = paddle.divide(a, b)
sign_ipt = paddle.sign(ipt)
abs_ipt = paddle.abs(ipt)
abs_ipt = paddle.floor(abs_ipt)
out = paddle.multiply(sign_ipt, abs_ipt)
# 输出
# Tensor(shape=[4], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [ 0.,  4., -0., -3.])
```
