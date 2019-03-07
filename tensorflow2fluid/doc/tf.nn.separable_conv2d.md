### tf.nn.depthwise_conv2d

#### [tf.nn.depthwise_conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)

```python
tf.nn.separable_conv2d(
    input,
    depthwise_filter,
    pointwise_filter,
    strides,
    padding,
    rate=None,
    name=None,
    data_format=None
)
```

#### [paddle.fluid.layers.conv2d](http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#paddle.fluid.layers.conv2d)

```python
paddle.fluid.layers.conv2d(
    input,
    num_filters,
    filter_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=None,
    param_attr=None,
    bias_attr=None,
    use_cudnn=True,
    act=None,
    name=None
)
```


#### 功能差异


#### 数据格式

TensorFlow: 默认且目前主流tensorflow模型的输入数据格式为`NHWC`，即表示`(batch，in_height, in_width, in_channels)`；
对应输出格式为`(batch, out_height, out_width, in_channel*channel_multiplier)`；卷积核的格式则为`(filter_height, filter_width, in_channels, channel_multiplier)`  
PaddlePaddle：输入数据格式为`NCHW`；输出格式`(batch, in_channel/groups, height, width)`；卷积核格式`(filters_num, in_channels, filter_height, filter_width)`

#### Padding机制
TensorFlow: `SAME`和`VALID`两种选项。当为`SAME`时，padding的计算方式如下所示
```python
# 计算在width上的padding size
# height上的padding计算方式同理
ceil_size = ceil(input_width / stride_width)
pad_size = (ceil_size - 1) * stride_width + filter_width - input_width
pad_left = ceil(pad_size / 2)
pad_right = pad_size - pad_left
```
PaddlePaddle：`padding`参数表示在输入图像四周padding的size大小
```
