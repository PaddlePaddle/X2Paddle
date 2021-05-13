## torch.nn.AvgPool2d
### [torch.nn.AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html?highlight=avgpool2d#torch.nn.AvgPool2d)

```python
torch.nn.AvgPool2d(kernel_size,
                   stride=None,
                   padding=0,
                   ceil_mode=False,
                   count_include_pad=True,
                   divisor_override=None)
```

### [paddle.nn.AvgPool2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/pooling/AvgPool2D_cn.html#avgpool2d)

```python
paddle.nn.AvgPool2D(kernel_size,
                    stride=None,
                    padding=0,
                    ceil_mode=False,
                    exclusive=True,
                    divisor_override=None,
                    data_format='NCHW',
                    name=None)
```

### 功能差异

#### 池化方式
***PyTorch***: 使用`count_include_pad`表示是否使用额外padding的值计算平均池化结果，默认为True。  
***PaddlePaddle***：使用`exclusive`表示是否不使用额外padding的值计算平均池化结果，默认为True。
