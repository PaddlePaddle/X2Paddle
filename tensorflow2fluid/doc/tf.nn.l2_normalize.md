
### tf.nn.l2_normalize

#### [tf.nn.l2_normalize](https://www.tensorflow.org/api_docs/python/tf/nn/l2_normalize)
``` python
tf.nn.l2_normalize(
    x,
    axis=None,
    epsilon=1e-12,
    name=None,
    dim=None
)
```

#### [paddle.fluid.layers.l2_normalize](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#l2-normalize)
``` python
layers.l2_normalize(
    x, 
    axis, 
    epsilon=1e-12, 
    name=None
)
```

#### 功能差异：
tensorflow：对于1-D tensor，axis为0时，output = x / sqrt(max(sum(x**2), epsilon))  

paddlepaddle：对于1-D tensor，axis为0时，output = x / sqrt(sum(x**2) + epsilon)

#### paddlepaddle示例:
```python
# 输入 tensor t 为[[1,2],[3,4]]

# 对t进行 l2 归一化
out = fluid.layers.l2_normalize(t, axis=1)  
```
