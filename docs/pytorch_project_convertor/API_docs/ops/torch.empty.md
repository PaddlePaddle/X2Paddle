## torch.empty
### [torch.empty](https://pytorch.org/docs/stable/generated/torch.empty.html?highlight=empty#torch.empty)

```python
torch.empty(*size,
            *,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False,
            pin_memory=False)
```

### [paddle.empty](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/empty_cn.html#empty)

```python
paddle.empty(shape,
             dtype=None,
             name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | shape        | 表示输出形状大小。                                     |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |
| pin_memeory   | -            | 表示是否使用锁页内存，PaddlePaddle无此参数。           |


### 功能差异

#### 使用方式
***PyTorch***：生成Tensor的形状大小以可变参数的方式传入。  
***PaddlePaddle***：生成Tensor的形状大小以list的方式传入。


### 代码示例
``` python
# PyTorch示例：
torch.empty(2, 3)
# 输出
# tensor([[9.1835e-41, 0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00]])
```

``` python
# PaddlePaddle示例：
paddle.empty([2, 3])
# 输出
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[0., 0., 0.],
#         [0., 0., 0.]])
```
