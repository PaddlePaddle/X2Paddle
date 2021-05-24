## torch.tensor
### [torch.from_numpy](https://pytorch.org/docs/stable/generated/torch.from_numpy.html?highlight=from_numpy#torch.from_numpy)

```python
torch.from_numpy(ndarray)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data,
                 dtype=None,
                 place=None,
                 stop_gradient=True)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| ndarray       | data        | 表示需要转换的数据。                                     |
| -             | dtype       | 表示数据类型，PyTorch无此参数。               |
| -        | place         | 表示Tensor存放位置，PyTorch无此参数。                   |
| -        | stop_gradient            | 表示是否阻断梯度传导，PyTorch无此参数。                   |

### 功能差异

#### 使用方式
***PyTorch***：只能传入一个numpy.ndarray。  
***PaddlePaddle***：可以传入scalar、list、tuple、numpy.ndarray、paddle.Tensor。
