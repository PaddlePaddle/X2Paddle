## torch.scatter
### [torch.scatter](https://pytorch.org/docs/stable/generated/torch.scatter.html?highlight=scatter#torch.scatter)

```python
torch.scatter(tensor,
            dim,
            index,
            src)
```

### [paddle.scatter_nd_add](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/scatter_nd_add_cn.html)

```python
paddle.scatter_nd_add(x,
             index,
             updates,
             name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | x        | 表示输入Tensor。                                     |
| dim           | -         | 表示在哪一个维度scatter,Paddle无此参数 |
| index        | index            | 输入的索引张量                   |
| src        | updates            | 输入的更新张量                   |



### 功能差异

#### 使用方式
因 torch.scatter 与 paddle.scatter_nd_add 差异较大，必须使用 paddle.flatten + paddle.meshgrid + paddle.scatter_nd_add 组合实现，看如下例子


### 代码示例
``` python
# PyTorch 示例：
src = torch.arange(1, 11).reshape((2, 5))
# 输出
# tensor([[ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10]])
index = torch.tensor([[0, 1, 2], [0, 1, 4]])
torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
# 输出
# tensor([[1, 2, 3, 0, 0],
#         [6, 7, 0, 0, 8],
#         [0, 0, 0, 0, 0]])
```

``` python
# PaddlePaddle 组合实现：
x = paddle.zeros([3, 5], dtype="int64")
updates = paddle.arange(1, 11).reshape([2,5])
# 输出
# Tensor(shape=[2, 5], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[1 , 2 , 3 , 4 , 5 ],
#         [6 , 7 , 8 , 9 , 10]])
index = paddle.to_tensor([[0, 1, 2], [0, 1, 4]])
i, j = index.shape
grid_x , grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
# 若 PyTorch 的 dim 取 0
# index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
# 若 PyTorch 的 dim 取 1
index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
# PaddlePaddle updates 的 shape 大小必须与 index 对应
updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
updates = paddle.gather_nd(updates, index=updates_index)
paddle.scatter_nd_add(x, index, updates)
# 输出
# Tensor(shape=[3, 5], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[1, 2, 3, 0, 0],
#         [6, 7, 0, 0, 8],
#         [0, 0, 0, 0, 0]])
```
