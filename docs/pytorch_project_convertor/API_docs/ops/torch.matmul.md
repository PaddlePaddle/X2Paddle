## torch.matmul
### [torch.matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html?highlight=matmul#torch.matmul)
```python
torch.matmul(input, other, out=None)
```

### [paddle.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/matmul_cn.html)
```python
paddle.matmul(x, y, transpose_x=False, transpose_y=False, name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x            | 表示输入的第一个Tensor。               |
| other        | y            | 表示输入的第二个Tensor。             |
| -           | transpose_x            | 表示相乘前是否转置x，PyTorch无此参数。               |
| -        | transpose_y            | 表示相乘前是否转置y，PyTorch无此参数。             |
| out          | -        | 表示输出的Tensor，PaddlePaddle无此参数。  |
