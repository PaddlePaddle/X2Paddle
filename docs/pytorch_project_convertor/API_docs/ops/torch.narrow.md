## torch.narrow
### [torch.narrow](https://pytorch.org/docs/stable/generated/torch.narrow.html?highlight=narrow#torch.narrow)
```python
torch.narrow(input, dim, start, length)
```
### [paddle.slice](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/slice_cn.html#slice)
```python
paddle.slice(input, axes, starts, ends)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim          | axes        | 表示切片的轴。                                     |
| start        | starts            | 表示起始位置。                   |
### 功能差异
#### 使用方式
***PyTorch***：只能在一个维度上进行切割，`dim`、`start`、`length`传入的值均只能为int型；使用该维度输出长度(`length`)来定位结束位置。  
***PaddlePaddle***：可以在多个维度进行切割，`axes`、`starts`、`ends`传入的值为list/tuple（`starts`、`ends`传入的值可以为tensor）；直接使用结束位置(`end`)来定位结束位置。

### 代码示例
``` python
# PyTorch示例：
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
torch.narrow(x, 0, 0, 2)
# 输出
# tensor([[ 1,  2,  3],
#         [ 4,  5,  6]])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
paddle.slice(x, [0], [0], [2])
# 输出
# Tensor(shape=[2, 3], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[1, 2, 3],
#         [4, 5, 6]])
```
