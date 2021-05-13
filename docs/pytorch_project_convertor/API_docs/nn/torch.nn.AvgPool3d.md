## torch.nn.AvgPool3d
### [torch.nn.AvgPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool3d.html?highlight=avgpool3d#torch.nn.AvgPool3d)

```python
torch.nn.AvgPool3d(kernel_size,
                   stride=None,
                   padding=0,
                   ceil_mode=False,
                   count_include_pad=True,
                   divisor_override=None)
```

### [paddle.nn.AvgPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/pooling/AvgPool3D_cn.html#avgpool3d)

```python
paddle.nn.AvgPool3D(kernel_size,
                    stride=None,
                    padding=0,
                    ceil_mode=False,
                    exclusive=True,
                    divisor_override=None,
                    data_format='NCDHW',
                    name=None)
```

### 功能差异

#### 池化方式
***PyTorch***: 使用`count_include_pad`表示是否使用额外padding的值计算平均池化结果，默认为True。  
***PaddlePaddle***：使用`exclusive`表示是否不使用额外padding的值计算平均池化结果，默认为True。
