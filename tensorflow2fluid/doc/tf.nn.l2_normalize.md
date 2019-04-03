
## tf.nn.l2_normalize

### [tf.nn.l2_normalize](https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize)

```python
tf.math.l2_normalize(
    x,
    axis=None,
    epsilon=1e-12,
    name=None,
    dim=None
)
```

### [paddle.fluid.layers.l2_normalize](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#l2-normalize)

```python
paddle.fluid.layers.l2_normalize(
    x, 
    axis, 
    epsilon=1e-12, 
    name=None
)
```

### 功能差异

#### 计算方式

TensorFlow：计算方式为`output = x / sqrt(max(sum(x^2), epsilon))`;  
PaddlePaddle：计算方式为`output = x / sqrt(sum(x^2) + epsilon))`。


### 代码示例
```
# x是shape为[3,2]的张量

# out同样是shape[3,2]的张量，axis设置为1，表示将x中每个行向量做归一化
out = fluid.layers.l2_normalize(x, axis=1)
```
