## torch.subtract
### [torch.subtract](https://pytorch.org/docs/stable/generated/torch.subtract.html?highlight=subtract#torch.subtract)
```python
torch.subtract(input, other, *, alpha=1, out=None)
```
### [paddle.subtract](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/subtract_cn.html#subtract)
```python
paddle.subtract(x, y, name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| alpha | -        | 表示`other`的乘数，PaddlePaddle无此参数。  |
| out          | -        | 表示输出的Tensor，PaddlePaddle无此参数。  |

### 功能差异

#### 计算差异
***PyTorch***：
$ out = input - alpha * other $

***PaddlePaddle***：
$ out = x - y $