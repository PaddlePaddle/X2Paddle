## torch.ones_like
### [torch.ones_like](https://pytorch.org/docs/stable/generated/torch.ones_like.html?highlight=ones_like#torch.ones_like)

```python
torch.ones_like(input, 
                *, 
                dtype=None, 
                layout=None, 
                device=None, 
                requires_grad=False, 
                memory_format=torch.preserve_format)
```

### [paddle.ones_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/ones_like_cn.html#ones-like)

```python
paddle.ones_like(x, dtype=None, name=None)
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
