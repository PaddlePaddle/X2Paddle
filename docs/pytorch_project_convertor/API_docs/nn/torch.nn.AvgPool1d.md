# torch.nn.AvgPool1d
### [torch.nn.AvgPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html?highlight=avgpool1d#torch.nn.AvgPool1d)

```python
torch.nn.AvgPool1d(kernel_size,
                   stride=None,
                   padding=0,
                   ceil_mode=False,
                   count_include_pad=True)
```

### [paddle.nn.AvgPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/pooling/AvgPool1D_cn.html#avgpool1d)

```python
paddle.nn.AvgPool1D(kernel_size,
                    stride=None,
                    padding=0,
                    exclusive=True,
                    ceil_mode=False,
                    name=None)
```

### 功能差异

#### 池化方式
***PyTorch***: 使用`count_include_pad`表示是否使用额外padding的值计算平均池化结果，默认为True。  
***PaddlePaddle***：使用`exclusive`表示是否不使用额外padding的值计算平均池化结果，默认为True。
