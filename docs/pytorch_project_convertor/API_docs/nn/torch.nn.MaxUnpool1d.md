## torch.nn.MaxUnpool1d
### [torch.nn.MaxUnpool1d](https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool1d.html?highlight=unpool#torch.nn.MaxUnpool1d)
```python
torch.nn.MaxUnpool1d(kernel_size, stride=None, padding=0)
```
### 功能介绍
用于实现一维反池化，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。
```python
import paddle
import paddle.nn as nn

# 定义MaxUnpool1D
class MaxUnpool1D(nn.Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(stride, int):
            self.kernel_size = [kernel_size]
        else:
            self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            if isinstance(stride, int):
                self.stride = [stride]
            else:
                self.stride = stride
        if isinstance(padding, int):
            self.padding = [padding]
        else:
            self.padding = padding

    def forward(self, input, indices, output_size=None):
        if output_size is None:
            n, c, l = input.shape
            out_l = (l - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            output_size = (n, c, out_l)
        else:
            if len(output_size) == len(self.kernel_size) + 2:
                output_size = output_size[2:]
        t = str(input.dtype).lower().strip().split(".")[-1]
        t = TYPE_MAPPER[t]
        out = paddle.zeros(output_size, dtype=t)
        flatten_out = paddle.flatten(out)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    indices[i, j, k] = (out.shape[1] * out.shape[2]) * i + out.shape[2] * j + indices[i, j, k]
        flatten_indices = paddle.flatten(indices)
        flatten_input = paddle.flatten(input)
        for i in range(flatten_indices.shape[0]):
            flatten_out[int(flatten_indices[i])] = flatten_input[i]
        out = paddle.reshape(flatten_out, out.shape)
        return out


# 组网
pool = nn.MaxPool1D(2, stride=2, return_mask=True)
unpool = MaxUnpool1D(2, stride=2)

# 构造输入
input = paddle.to_tensor([[[ 1.,  2,  3,  4]]])

# 进行池化
pool_res, indices = pool(input)
# pool_res:
# Tensor(shape=[1, 1, 2], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[2., 4.]]])
# indices:
# Tensor(shape=[1, 1, 2], dtype=int32, place=CPUPlace, stop_gradient=True,
#        [[[1, 3]]])

# 进行反池化
res = unpool(pool_res, indices)
# res:
# Tensor(shape=[1, 1, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[0., 2., 0., 4.]]])
```
