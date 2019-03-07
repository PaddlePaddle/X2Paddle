
## tf.layers.conv2d

### [tf.image.resize_bilinear](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)
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

TensorFlow: 默认且目前主流tensorflow模型的输入数据格式为`NHWC`，即表示`(batch`，height, width, in_channel)`；
对应输出格式为`(batch, height, width, filter_num)`;
PaddlePaddle：输入数据格式为`NCHW`，


## paddlepaddle示例:
```python

# 常量tensor out 中数据为 np.array([[5,5,5],[5,5,5]], dtype='int64')
out = fluid.layers.fill_constant(shape=[2,3], dtype='int64', value=5)  
