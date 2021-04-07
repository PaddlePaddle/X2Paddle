## torch.tensor
### [torch.from_numpy](https://pytorch.org/docs/stable/generated/torch.from_numpy.html?highlight=from_numpy#torch.from_numpy)

```python
torch.from_numpy(ndarray)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data, 
                 dtype=None, 
                 place=None, 
                 stop_gradient=True)
```

### 功能差异

#### 使用方式
PyTorch：只能传入一个numpy.ndarray。
PaddlePaddle：可以传入scalar、list、tuple、numpy.ndarray、paddle.Tensor。

#### 参数使用
#### 设置数据类型
PyTorch：无此功能。  
PaddlePaddle：`dtype`表示数据类型。 
#### 设置设备位置
PyTorch：无此功能。  
PaddlePaddle：`place`表示设备位置。  
#### 设置梯度反传
PyTorch：无此功能。  
PaddlePaddle：`stop_gradient`表示是否阻断Autograd的梯度传导。 