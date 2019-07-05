## tf.contrib.layers.flatten

### [tf.contrib.layers.flatten](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/contrib/layers/flatten)

```python
tf.contrib.layers.flatten(
    inputs,
    outputs_collections=None,
    scope=None
)
```

### [paddle.fluid.layers.flatten](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#flatten)

```python
paddle.fluid.layers.flatten(
    x, 
    axis=1, 
    name=None
)
```

### 功能差异

#### 计算方式

TensorFlow：固定第0维，将其他维合并；  

PaddlePaddle：使用`axis`指定两次合并的维度边界，参考下面示例。

### 代码示例
```
# 张量x的shape为 [2, 3, 4, 5]
out = fluid.layers.flatten(x, axis=2)
out.shape # [2*3, 4*5]

```