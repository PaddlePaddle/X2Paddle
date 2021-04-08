## torch.empty_like
### [torch.empty_like](https://pytorch.org/docs/stable/generated/torch.empty_like.html?highlight=empty_like#torch.empty_like)

```python
torch.empty_like(input, 
                 *, 
                 dtype=None, 
                 layout=None, 
                 device=None, 
                 requires_grad=False, 
                 memory_format=torch.preserve_format)
```

### [paddle.empty_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/empty_like_cn.html#empty-like)

```python
paddle.empty_like(x, dtype=None, name=None)
```

### 功能差异
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
#### 设置内存格式
PyTorch：`memory_format`表示内存格式。  
PaddlePaddle：无此功能。  

