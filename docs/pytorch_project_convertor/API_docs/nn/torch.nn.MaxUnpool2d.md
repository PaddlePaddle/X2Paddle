## torch.nn.MaxUnpool2d
### [torch.nn.MaxUnpool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html?highlight=maxunpool2d#torch.nn.MaxUnpool2d)
```python
torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
```
### 功能介绍
用于实现一维反池化，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。

```python
import paddle
import paddle.nn as nn
TYPE_MAPPER = {"fp16": "float16", "fp32": "float32", "fp64": "float64"}

# 定义MaxUnpool2D
class MaxUnpool2D(paddle.nn.Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(stride, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            if isinstance(stride, int):
                self.stride = (stride, stride)
            else:
                self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def forward(self, input, indices, output_size=None):
        if output_size is None:
            n, c, h, w = input.shape
            out_h = (h - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            out_w = (w - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
            output_size = (n, c, out_h, out_w)
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
                    for m in range(indices.shape[3]):
                        indices[i, j, k, m] = (out.shape[1] * out.shape[2] * out.shape[3]) * i + \
                                              (out.shape[2] * out.shape[3]) * j + indices[i, j, k, m]
        flatten_indices = paddle.flatten(indices)
        flatten_input = paddle.flatten(input)
        for i in range(flatten_indices.shape[0]):
            flatten_out[flatten_indices[i].tolist()] = flatten_input[i].tolist()
        out = paddle.reshape(flatten_out, out.shape)
        return out


# 组网
pool = nn.MaxPool2D(2, stride=2, return_mask=True)
unpool = MaxUnpool2D(2, stride=2)

# 构造输入
input = paddle.to_tensor([[[[ 1.,  2,  3,  4],
                            [ 5,  6,  7,  8],
                            [ 9, 10, 11, 12],
                            [13, 14, 15, 16]]]])

# 进行池化
pool_res, indices = pool(input)
# pool_res:
# Tensor(shape=[1, 1, 2, 2], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[[6. , 8. ],
#           [14., 16.]]]])
# indices:
# Tensor(shape=[1, 1, 2, 2], dtype=int32, place=CPUPlace, stop_gradient=True,
#        [[[[5 , 7 ],
#           [13, 15]]]])

# 进行反池化
res = unpool(pool_res, indices)
# res:
# Tensor(shape=[1, 1, 4, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[[0. , 0. , 0. , 0. ],
#           [0. , 6. , 0. , 8. ],
#           [0. , 0. , 0. , 0. ],
#           [0. , 14., 0. , 16.]]]])
```
