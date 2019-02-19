### tf.split

#### [tf.split](https://www.tensorflow.org/api_docs/python/tf/split)

```python
tf.split(
    value,
    num_or_size_splits,
    axis=0,
    num=None,
    name='split'
)
```

#### [paddle.fluid.layers.conv2d](http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#paddle.fluid.layers.conv2d)

```python
paddle.fluid.layers.split(
    input, 
    num_or_sections, 
    dim=-1, 
    name=None
)
```

#### 功能差异

1. 返回值

##### 1. 返回值

&#160; &#160; &#160; &#160;Tensorflow中，`split`函数返回的结果均保存在一个tensor类型的值中；而在PaddlePaddle中，`split`返回`list`类型结果，长度为`num_or_sections`。
