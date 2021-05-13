## torch.narrow
### [torch.narrow](https://pytorch.org/docs/stable/generated/torch.narrow.html?highlight=narrow#torch.narrow)
```python
torch.narrow(input, dim, start, length)
```

### [paddle.slice](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/layers/slice_cn.html#slice)
```python
paddle.slice(input, axes, starts, ends)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim          | axes        | 表示切片的轴。                                     |
| start        | starts            | 表示起始位置。                   |
