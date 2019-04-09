## tf.split

### [tf.split](https://www.tensorflow.org/api_docs/python/tf/split)

```python
tf.split(
    value,
    num_or_size_splits,
    axis=0,
    num=None,
    name='split'
)
```

### [paddle.fluid.layers.split](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#split)

```python
paddle.fluid.layers.split(
    input, 
    num_or_sections, 
    dim=-1, 
    name=None
)
```

### 功能差异

#### 返回值类型

TensorFlow：`split`函数返回的结果均保存在一个tensor类型的值中;  

PaddlePaddle：`split`返回`list`类型结果，长度为`num_or_sections`。

### 代码示例
```
# x是shape为[3,9,5]的张量：
x0, x1, x2 = fluid.layers.split(x, num_or_sections=3, dim=1)
x0.shape  # [3, 3, 5]
x1.shape  # [3, 3, 5]
x2.shape  # [3, 3, 5]

```
