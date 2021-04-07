## torch.zeros
### [torch.zeros](https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=zeros_like#torch.zeros_like)

```python
torch.zeros_like(input, 
                 *, 
                 dtype=None, 
                 layout=None, 
                 device=None, 
                 requires_grad=False, 
                 memory_format=torch.preserve_format)
```

### [paddle.zeros](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/zeros_like_cn.html#zeros-like)

```python
paddle.zeros_like(x, dtype=None, name=None)
```

### 功能差异
#### 参数使用
#### 设置设备位置
PyTorch：`device`表示设备位置。  
PaddlePaddle：无此功能。  
#### 设置梯度反传
PyTorch：`requires_grad`表示是否阻断Autograd的梯度传导。  
PaddlePaddle：无此功能。  

