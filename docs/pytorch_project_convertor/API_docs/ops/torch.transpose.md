## torch.transpose
### [torch.transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html?highlight=transpose#torch.transpose)

```python
torch.transpose(input, dim0, dim1)
```

### [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/transpose_cn.html#transpose)

```python
paddle.transpose(x, perm, name=None)
```

### 功能差异
#### 使用方式
***PyTorch***：需要设置2个维度值(`dim0`和`dim1`)表示需要交换的维度。  
***PaddlePaddle***：需要设置一个重排顺序(`perm`)，类型为list或者tuple。

### 代码示例
``` python
# PyTorch示例：
x = torch.ones((10,20,30))
out = torch.transpose(x, 0, 2)
out.shape
# 输出
# torch.Size([30, 20, 10])
```

``` python
# PaddlePaddle示例：
x = paddle.ones((10,20,30))
out = paddle.transpose(x, (2, 1, 0))
out.shape
# 输出
# [30, 20, 10]
```
