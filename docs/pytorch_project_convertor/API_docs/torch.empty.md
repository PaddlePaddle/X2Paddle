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

### [paddle.empty](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/empty_cn.html#empty)

```python
paddle.empty(shape, 
             dtype=None, 
             name=None)
```

### 功能差异

#### 使用方式
PyTorch：生成Tensor的形状大小以可变参数的方式传入。   
PaddlePaddle：生成Tensor的形状大小以list的方式传入。

#### 参数使用
#### 设置布局方式
PyTorch：`layout`表示布局方式。  
PaddlePaddle：无此功能。  
#### 设置设备位置
PyTorch：`device`表示设备位置。  
PaddlePaddle：无此功能。  
#### 设置梯度反传
PyTorch：`requires_grad`表示是否阻断Autograd的梯度传导。  
PaddlePaddle：无此功能。 
#### 设置锁页内存
PyTorch：`pin_memeory`表示是否使用锁页内存。  
PaddlePaddle：无此功能。  

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
