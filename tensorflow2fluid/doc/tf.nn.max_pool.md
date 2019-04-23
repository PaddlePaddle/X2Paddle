## tf.nn.max_pool

### [tf.nn.max_pool](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)

``` python
tf.nn.max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)
```


### [paddle.fluid.layers.pool2d](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#paddle.fluid.layers.pool2d)
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

Tensorflow: 存在`SAME`和`VALID`两种padding方式。当为`SAME`时，padding的size计算方式如下，仅在最右和最下进行padding；
```
ceil_size = ceil(input_size / stride)
pad_size = (ceil_size - 1) * stride + filter_size - input_size
```
PaddlePaddle：在输入的上、下、左、右分别padding，size大小为`pool_padding`，通过示例代码，可实现与Tensorflow中`max_pool`的`SAME`方式。

### 代码示例
```
inputs = fluid.layers.data(dtype='float32', shape=[3, 300, 300], name='inputs')

# 计算得到输入的长、宽对应padding size为1
# 在最右、最下进行padding
pad_res = fluid.layers.pad2d(inputs, padding=[0, 1, 0, 1])
conv_res = fluid.layers.pool2d(pad_res, pool_size=3, pool_type='max', pool_stride=2)
```
