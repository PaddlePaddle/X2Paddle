### tf.math.maximum

#### [tf.math.maximum](https://www.tensorflow.org/api_docs/python/tf/math/maximum)

```python
tf.math.maximum(
    x,
    y,
    name=None
)
```

#### [paddle.fluid.layers.conv2d](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-63-elementwise_max)

```python
paddle.fluid.layers.elementwise_max(
    x, 
    y, 
    axis=-1, 
    act=None, 
    name=None)
```

#### 功能差异

##### 1. 广播Broadcast机制
由于广播broadcast机制的不同，在paddlepaddle中，`y`的`shape`必须与`x`的`shape`相同，或者是其连续子序列
