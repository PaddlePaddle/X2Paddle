## torch.tensor
### [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html?highlight=tensor#torch.tensor)

```python
torch.tensor(data, 
             *, 
             dtype=None, 
             device=None, 
             requires_grad=False, 
             pin_memory=False) 
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data, 
                 dtype=None, 
                 place=None, 
                 stop_gradient=True)
```

### 功能差异

#### 参数使用
#### 设置设备位置
PyTorch：`device`表示设备位置。  
PaddlePaddle：`place`表示设备位置。  
#### 设置梯度反传
PyTorch：`requires_grad`表示是否阻断Autograd的梯度传导，默认值为False，代表不进行梯度传导。  
PaddlePaddle：`stop_gradient`表示是否阻断Autograd的梯度传导，默认值为True，代表不进行梯度传导。  
#### 设置锁页内存
PyTorch：`pin_memeory`表示是否使用锁页内存。  
PaddlePaddle：无此功能。  


