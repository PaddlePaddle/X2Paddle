
## tf.nn.conv3d_transpose

### [tf.nn.conv3d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv3d_transpose)
``` python
tf.nn.conv3d_transpose(
    value,
    filter,
    output_shape,
    strides,
    padding='SAME',
    data_format='NDHWC',
    name=None
)
```

### [paddle.fluid.layers.conv3d_transpose](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-48-conv3d_transpose)
``` python
paddle.fluid.layers.conv3d_transpose(
    input, 
    num_filters, 
    output_size=None, 
    filter_size=None, 
    padding=0, 
    stride=1, 
    dilation=1, 
    groups=None, 
    param_attr=None, 
    bias_attr=None, 
    use_cudnn=True, 
    act=None, 
    name=None
)
```

### 功能差异

#### 数据格式

TensorFlow: 默认输入数据格式为`NDHWC`，表示`(batch，depth, height, width, in_channels)`， 同时也将`data_format`参数设为`channels_first`，支持`NCDHW`格式的数据输入。其中输入、输出、卷积核对应关系如下表所示，

| 输入 | 卷积核 | 输出 |
|--------------------|-------------------|------------------|
|NDHWC | (kernel_d, kernel_h, kernel_w, filters_num, in_channels)| (batch, out_d, out_h, out_w, filters_num)|
|NCDHW | (kernel_d, kernel_h, kernel_w, filters_num, in_channels) | (batch, filters_num, out_d, out_h, out_w)|

PaddlePaddle: 只支持输入数据格式为`NCDHW`，且**卷积核格式**与TensorFlow不同，其中输入、输出、卷积核对应关系如下表所示，

| 输入 | 卷积核 | 输出 |
|--------------------|-------------------|------------------|
|NCDHW | (in_channels, filters_num, kernel_d, kernel_h, kernel_w) | (batch, filters_num, out_d, out_h, out_w)|

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

#### 输出大小
TensorFlow：当padding为`SAME`和`VALID`两种情况下，输出大小计算方式如下所示
```python
if padding == 'SAME':
    output_size = input_size * stride
elif padding == 'VALID':
    output_size = input_size * stride + max(kernel_size - stride, 0)
```
PaddlePaddle: 输出大小计算公式如下，差异主要由于TensorFlow在`conv2d_transpose`的最后还存在**裁剪**步骤，因此可参考示例代码，调用`crop`解决
```python
output_size = (input_size - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1
```

### 代码示例
```python
# TensorFlow使用conv3d_transpose
# 输入shape: [-1, 5， 20, 40, 3]
inputs = tf.placeholder(dtype=tf.float32, shape=[None, 5， 20, 40, 3])
filter = tf.random_uniform(shape=[2, 4, 5, 7， 3], 0.0， 1.0)
batch = tf.shape(inputs)[0]
# conv2d_transpose输出shape: [-1, 5, 40, 80， 7]
result = tf.nn.conv2d_transpose(inputs, filter, output_shape=[batch, 5, 40, 80， 7], 
                         strides=(1, 2, 2), padding='SAME')

#PaddlePaddle中使用conv3d_transpose
# 输入Shape：(None, 3, 5, 20, 40)
inputs = fluid.layers.data(dtype='float32', shape=[3, 5, 20, 40], name='inputs)
# conv3d_transpose输出shape:[-1, 7, 6, 40, 81]
outputs = fluid.layers.conv3d(inputs, 7, filter_size=(2, 4, 5), stride=(1, 2, 2), 
                        padding=(0, 1, 1), bias_attr=False)
# 裁剪后结果即为与TensorFlow一致
outputs = fluid.layers.crop(outputs, shape=[-1, 7, 5, 40, 80])
