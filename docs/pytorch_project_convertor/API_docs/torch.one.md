## torch.ones
### [torch.ones](https://pytorch.org/docs/stable/generated/torch.ones.html?highlight=ones#torch.ones)

```python
torch.ones(*size, 
           *, 
           out=None, 
           dtype=None, 
           layout=torch.strided, 
           device=None, 
           requires_grad=False)
```

### [paddle.ones](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/ones_cn.html#ones)

```python
paddle.ones(shape, 
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

### 代码示例
``` python
# PyTorch示例：
torch.ones(2, 3)
# 输出
# tensor([[ 1.,  1.,  1.],
#         [ 1.,  1.,  1.]])
```

``` python
# PaddlePaddle示例：
paddle.ones([2, 3])
# 输出
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[1., 1., 1.],
#         [1., 1., 1.]])
```
