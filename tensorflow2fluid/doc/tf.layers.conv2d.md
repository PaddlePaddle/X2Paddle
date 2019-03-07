
## tf.layers.conv2d

### [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)
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

### [paddle.fluid.layers.conv2d](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#paddle.fluid.layers.conv2d)
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

### 功能差异：

#### 数据格式

TensorFlow: 默认且目前主流tensorflow模型的输入数据格式为`NHWC`，即表示`(batch，height, width, in_channels)`；
对应输出格式为`(batch, height, width, filters_num)`；卷积核的格式则为`(filter_height, filter_width, in_channels, filters_num)`  
PaddlePaddle：输入数据格式为`NCHW`；输出格式`(batch, filters_num, height, width)`；卷积核格式`(filters_num, in_channels, filter_height, filter_width)`

#### Padding机制
TensorFlow: `SAME`和`VALID`两种选项。当为`SAME`时，padding的计算方式如下
PaddlePaddle：`padding`参数表示在输入图像四周padding的size大小

#### 参数差异
TensorFlow：深度可分离卷积使用[tf.layers.separable_conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/separable_conv2d)接口
PaddlePaddle: 使用`paddle.fluid.layers.conv2d`，可参考
[PaddlePaddle对卷积的说明文档](http://paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/layers/conv.html), 同时也可参考[tf.nn.depthwise_conv2d](https://github.com/PaddlePaddle/X2Paddle/blob/doc/tensorflow2fluid/doc/tf.nn.conv2d)中的代码示例。

## paddlepaddle示例:
```python
# 示例1 
# 结合pad2d，实现SAME方式的padding
# 输入Shape：(None, 3, 200, 200)
# 输出Shape：(None, 5， 67， 67）
# 卷积核Shape: (5, 3, 4, 4)
inputs = paddle.fluid.layers.data(dtype='float32', shape=[3, 200, 200], name='inputs)
pad_inputs = paddle.fluid.layers.pad2d(inputs, paddings=[1, 2, 1, 2])
outputs = paddle.fluid.layers.conv2d(pad_inputs, 5, [4, 4], (1, 1))
