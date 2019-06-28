## tf.nn.conv2d

### [tf.nn.conv2d](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/nn/conv2d)

```python
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```

### [paddle.fluid.layers.conv2d](http://www.paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#paddle.fluid.layers.conv2d)

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

### 功能差异

`tf.nn.conv2d`中的参数`filter`为具体的tensor，而`paddle.fluid.layers.conv2d`参数中则声明卷积核的`size`，函数内部创建卷积核tensor。也可通过如下代码示例，自行创建并复用卷积核  
需要注意的是PaddlePaddle中的输入、输出以及卷积核的格式与tensorflow存在部分差异，可参考[tf.layers.conv2d](https://github.com/PaddlePaddle/X2Paddle/blob/master/tensorflow2fluid/doc/tf.layers.conv2d.md)

### 代码示例  
```python
# 输入为NCHW格式
inputs = fluid.layers.data(dtype='float32', shape=[-1, 3, 300, 300], name='inputs')
create_kernel = fluid.layers.create_parameters(shape=[5, 3, 2, 2], dtype='float32', name='kernel')

# PaddlePaddle中可通过相同的参数命名引用同一个参数变量
# 通过指定卷积核参数名(param_attr)为'kernel'，引用了create_kernel
result = fluid.layers.conv2d(inputs, 5, [2, 2], param_attr='kernel')
```