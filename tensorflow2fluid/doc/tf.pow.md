## tf.pow

### [tf.pow](https://www.tensorflow.org/api_docs/python/tf/math/pow)

```python
tf.math.pow(
    x,
    y,
    name=None
)
```

### [paddle.fluid.layers.pow](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#pow)

```python
paddle.fluid.layers.pow(
    x, 
    factor=1.0, 
    name=None
)
```

### 功能差异

#### 参数类型

TensorFlow：`x`与`y`为shape相同的tensor，执行element-wise求幂操作；  

PaddlePaddle：`x`为tensor，`factor`为浮点数，返回值为`x`每个元素执行按照`factor`执行求幂操作得到的tensor。

### 代码示例
```
# x为张量 [2, 3]
out = fluid.layers.pow(x, 2.0) # [4，9]

```
