## tf.nn.avg_pool

### [tf.nn.avg_pool](https://www.tensorflow.org/versions/r1.10/api_docs/python/tf/nn/avg_pool)

``` python
tf.nn.avg_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)
```


### [paddle.fluid.layers.pool2d](http://paddlepaddle.org/documentation/docs/en/1.3/api/layers.html#permalink-116-pool2d)
``` python
paddle.fluid.layers.pool2d(
    input,
    pool_size=-1,
    pool_type='max',
    pool_stride=1,
    pool_padding=0,
    global_pooling=False,
    use_cudnn=True,
    ceil_mode=False,
    name=None,
    exclusive=True)
```
### 功能差异

#### 输入格式
TensorFlow: 默认为`NHWC`的数据输入格式，同时也可通过修改`data_format`参数，支持`NCHW`的输入；
PaddlePaddle：只支持`NCHW`的数据输入格式。

#### Padding机制

Tensorflow: 存在`SAME`和`VALID`两种padding方式。当为`SAME`时，padding的size计算方式如下伪代码所示，需要注意的是，当计算得到的`pad_size`为奇
数时，右侧与下方相对比左侧和上方会多1个size；
``` python
# 计算在width上的padding size
# height上的padding计算方式同理
ceil_size = ceil(input_width / stride_width)
pad_size = (ceil_size - 1) * stride_width + filter_width - input_width
pad_left = ceil(pad_size / 2)
pad_right = pad_size - pad_left
```
PaddlePaddle：在输入的上、下、左、右分别padding，size大小为`pool_padding`。

### 代码示例
```
inputs = fluid.layers.data(dtype='float32', shape=[3, 300, 300], name='inputs')

# 计算得到输入的长、宽对应padding size为1
# 当Tensorflow中padding为SAME时，可能会两侧padding的size不同，可调用pad2d对齐
pad_res = fluid.layers.pad2d(inputs, paddings=[0, 1, 0, 1])
conv_res = fluid.layers.pool2d(pad_res, pool_size=3, pool_type='avg', padding=[1, 1], pool_stride=2)
```
