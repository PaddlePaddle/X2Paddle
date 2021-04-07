## torch.zeros
### [torch.zeros](https://pytorch.org/docs/stable/generated/torch.zeros.html?highlight=zeros#torch.zeros)

```python
torch.zeros(*size, 
            *, 
            out=None, 
            dtype=None, 
            layout=torch.strided, 
            device=None, 
            requires_grad=False)
```

### [paddle.zeros](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/zeros_cn.html#zeros)

```python
paddle.zeros(shape, 
             dtype=None, 
             name=None)
```

### 功能差异

#### 使用方式
PyTorch：生成Tensor的形状大小以可变参数的方式传入。
PaddlePaddle：生成Tensor的形状大小以list的方式传入。

#### 参数使用
#### 设置数据类型
PyTorch：`dtype`表示数据类型。  
PaddlePaddle：无此功能。  
#### 设置设备位置
PyTorch：`device`表示设备位置。  
PaddlePaddle：无此功能。  
#### 设置梯度反传
PyTorch：`requires_grad`表示是否阻断Autograd的梯度传导。  
PaddlePaddle：无此功能。  

### 代码示例
``` python
# PyTorch示例：
torch.zeros(2, 3)
# 输出
# tensor([[ 0.,  0.,  0.],
#         [ 0.,  0.,  0.]])
```

``` python
# PaddlePaddle示例：
paddle.zeros([2, 3])
# 输出
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[0., 0., 0.],
#         [0., 0., 0.]])
```
