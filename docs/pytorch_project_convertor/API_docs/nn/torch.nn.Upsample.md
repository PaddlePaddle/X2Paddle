## torch.nn.Upsample
### [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html?highlight=upsample#torch.nn.Upsample)
```python
torch.nn.Upsample(size=None,
                  scale_factor=None,
                  mode='nearest',
                  align_corners=False)
```
### [paddle.nn.Upsample](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Upsample_cn.html#upsample)
```python
paddle.nn.Upsample(size=None,
                   scale_factor=None,
                   mode='nearest',
                   align_corners=False,
                   align_mode=0,
                   data_format='NCHW',
                   name=None)
```

### 功能差异

#### 输入格式
***PyTorch***：只支持`NCHW`的输入。
***PaddlePaddle***：支持`NCHW`和`NHWC`两种格式的输入（通过`data_format`设置）。

#### 计算方式
***PyTorch***：在mode为`bilinear`或`trilinear`时，只支持align_mode为0的上采样。
***PaddlePaddle***：在mode为`bilinear`或`trilinear`时，支持align_mode为0和1的上采样。
【注意】align_mode为0或1时的上采样方式可参见[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/common/Upsample_cn.html#upsample)。
