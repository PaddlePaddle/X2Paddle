
## tf.nn.top_k

### [tf.nn.top_k](https://www.tensorflow.org/api_docs/python/tf/nn/top_k)
``` python
tf.math.top_k(
    input,
    k=1,
    sorted=True,
    name=None
)
```

### [paddle.fluid.layers.topk](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#topk)
``` python
paddle.fluid.layers.topk(
    input, 
    k, 
    name=None
)
```

### 功能差异
#### 参数差异
TensorFlow: 通过设置`sorted`参数，对返回的值与下标设置是否进行降序排序；`k`默认为1。  
PaddlePaddle: 对返回的top-k tensor进行降序排序；`k`没有默认值，必须设置。

### 代码示例
```python
# 输入 tensor t 为[[2,6,3],[3,0,8]]

# 当k=2时，输出 tensor out 为[[6,3], [8,3]]，index为[[1,2],[2,0]]
out, index = fluid.layers.topk(t, k=1)

```
