### tf.pow

#### [tf.pow](https://www.tensorflow.org/api_docs/python/tf/pow)

```python
tf.pow(
    x, 
    y, 
    name=None
)
```

#### [paddle.fluid.layers.pow](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#pow)

```python
paddle.fluid.layers.pow(
    x, 
    factor=1.0, 
    name=None
)
```

#### 功能差异

##### 参数类型

Tensorflow：`x`作为基数，`y`作为指数，要求二者可以broadcast，实现elementwise的幂运算;  
PaddlePaddle：`factor`作为指数，要求必须是python scalar，不涉及broadcast。

#### paddlepaddle代码示例
```
# x是张量np.array([1,2,3])

# factor=2时，张量out为np.array([1,4,9])
out = fluid.layers.pow(x, factor=2)

```
