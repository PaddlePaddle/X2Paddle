## torch.tensor
### [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html?highlight=tensor#torch.tensor)

```python
torch.tensor(data,
             dtype=None,
             device=None,
             requires_grad=False,
             pin_memory=False)
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
| device        | place            | 表示Tensor存放位置。                   |
| requires_grad | stop_gradient    | PyTorch表示是否不阻断梯度传导，PaddlePaddle表示是否阻断梯度传导。 |
| pin_memeory   | -            | 表示是否使用锁页内存，PaddlePaddle无此参数。           |  
