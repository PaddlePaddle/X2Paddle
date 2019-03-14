
### tf.nn.l2_normalize

#### [tf.nn.l2_normalize](https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize)

```python
tf.math.l2_normalize(
    x,
    axis=None,
    epsilon=1e-12,
    name=None,
    dim=None
)
```

#### [paddle.fluid.layers.l2_normalize](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#l2-normalize)

```python
paddle.fluid.layers.l2_normalize(
    x, 
    axis, 
    epsilon=1e-12, 
    name=None
)
```

#### 功能差异

##### 计算公式

Tensorflow：公式为output = x / sqrt(max(sum(x^2), epsilon));  
PaddlePaddle：公式为output = x / sqrt(sum(x^2) + epsilon))。

##### 参数类型

Tensorflow：`axis`参数可以是scalar和list；  
PaddlePaddle：`axis`参数只能是scalar。

#### paddlepaddle代码示例
```
# x是shape为[3,2]的张量

# out同样是shape[3,2]的张量，axis设置为1，表示将x中每个行向量做归一化
out = fluid.layers.l2_normalize(x, axis=1)


```
