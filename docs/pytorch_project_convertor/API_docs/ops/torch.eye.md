## torch.eye

### [torch.eye](https://pytorch.org/docs/stable/generated/torch.eye.html?highlight=eye#torch.eye)
```python
torch.eye(n,
          m=None,
          *,
          out=None,
          dtype=None,
          layout=torch.strided,
          device=None,
          requires_grad=False)
```

### [paddle.eye](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/eye_cn.html#eye)
```python
paddle.eye(num_rows,
           num_columns=None,
           dtype=None,
           name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| n           | num_rows            | 生成2-D Tensor的行数。               |
| m        | num_columns            | 生成2-D Tensor的列数。                   |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |
