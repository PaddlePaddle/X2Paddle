## tf.nn.separable_conv2d

### [tf.nn.separable_conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)
``` python
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

### PaddlePaddle实现
PaddlePaddle中目前无对应接口，可使用如下代码实现，在如下代码中只考虑了基本的`strides`参数，其它参数如`padding`在PaddlePaddle中使用机制
以及输入输出和卷积核格式与TensorFlow存在差异，可参考文档[tf.layers.conv2d](https://github.com/PaddlePaddle/X2Paddle/blob/master/tensorflow2fluid/doc/tf.layers.conv2d.md)和[tf.nn.depthwise_conv2d](https://github.com/PaddlePaddle/X2Paddle/blob/master/tensorflow2fluid/doc/tf.nn.depthwise_conv2d.md)中的说明。
``` python
# TensorFlow中separable_conv2d的使用
depthwise_filter = tf.random_uniform([4, 4, 3, 1], 0.0, 1.0)
pointwise_filter = tf.random_uniform([1, 1, 3, 5], 0.0, 1.0)
result = tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, 
                        strides=[1, 1, 1, 1], padding='VALID')

# PaddlePaddle中对应如上代码实现separable_conv2d
depthwise_result = fluid.layers.conv2d(input, 3, filter_size=[4, 4], 
                                stride=[1, 1], groups=3, bias_attr=False)
pointwise_result = fluid.layers.conv2d(depthwise_result, filter_size=[1, 1], 
                                stride=[1, 1], bias_attr=False)

```
