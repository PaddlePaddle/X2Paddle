## tf.layers.conv2d

### [tf.layers.conv2d](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/layers/conv2d)
``` python
tf.layers.conv2d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
```

### [paddle.fluid.layers.conv2d](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#paddle.fluid.layers.conv2d)
``` python
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
    name=None)
```

### 功能差异

#### 数据格式

TensorFlow: 默认输入数据格式为`NHWC`，表示`(batch，height, width, in_channels)`， 同时也将`data_format`参数设为`channels_first`，支持`NCHW`格式的数据输入。其中输入、输出、卷积核对应关系如下表所示，

| 输入 | 卷积核 | 输出 |
|--------------------|-------------------|------------------|
|NHWC | (kernel_h, kernel_w, filters_num, in_channels)| (batch, out_h, out_w, filters_num)|
|NCHW | (kernel_h, kernel_w, filters_num, in_channels) | (batch, filters_num, out_h, out_w)|

PaddlePaddle：只支持输入数据格式为`NCHW`，且**卷积核格式**与TensorFlow不同，其中输入、输出、卷积核对应关系如下表所示，

| 输入 | 卷积核 | 输出 |
|--------------------|-------------------|------------------|
|NCHW | (in_channels, filters_num, kernel_h, kernel_w) | (batch, filters_num, out_h, out_w)|

#### Padding机制
TensorFlow: `SAME`和`VALID`两种选项。当为`SAME`时，padding的计算方式如下所示,
```python
# 计算在width上的padding size
# height上的padding计算方式同理
ceil_size = ceil(input_width / stride_width)
pad_size = (ceil_size - 1) * stride_width + filter_width - input_width
pad_left = ceil(pad_size / 2)
pad_right = pad_size - pad_left
```
PaddlePaddle：`padding`参数表示在输入图像四周padding的size大小。

#### 参数差异
TensorFlow：深度可分离卷积使用[tf.layers.separable_conv2d](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/layers/separable_conv2d)接口;  
PaddlePaddle: 使用`paddle.fluid.layers.conv2d`，可参考
[PaddlePaddle对卷积的说明文档](http://paddlepaddle.org/documentation/docs/zh/1.4/api_guides/low_level/layers/conv.html), 同时也可参考[tf.nn.separable_conv2d](https://github.com/PaddlePaddle/X2Paddle/blob/master/tensorflow2fluid/doc/tf.nn.separable_conv2d.md)中的代码示例。

### 代码示例
```python
# 结合pad2d，实现SAME方式的padding
# 输入Shape：(None, 3, 200, 200)
# 输出Shape：(None, 5， 67， 67）
# 卷积核Shape: (5, 3, 4, 4)
inputs = paddle.fluid.layers.data(dtype='float32', shape=[3, 200, 200], name='inputs)
pad_inputs = paddle.fluid.layers.pad2d(inputs, paddings=[1, 2, 1, 2])
outputs = paddle.fluid.layers.conv2d(pad_inputs, 5, [4, 4], (1, 1))
