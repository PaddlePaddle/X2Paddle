
## tf.nn.lrn

### [tf.nn.lrn](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization)

```python
tf.nn.local_response_normalization(
    input,
    depth_radius=5,
    bias=1,
    alpha=1,
    beta=0.5,
    name=None
)
```

### [paddle.fluid.layers.lrn](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#paddle.fluid.layers.lrn)

```python
paddle.fluid.layers.lrn(
    input, 
    n=5, 
    k=1.0, 
    alpha=0.0001, 
    beta=0.75, 
    name=None
)
```

### 功能差异

#### 计算方式

TensorFlow：计算公式如下所示，公式中的$n$即为参数`depth_radius`
$$output(i,x,y)=input(i,x,y)/(k+\alpha\sum_{j=max(0,i-n)}^{min(C,i+n+1)}{input(j,x,y)^2})^\beta$$ 
PaddlePaddle：计算公式如下所示，
$$output(i,x,y)=input(i,x,y)/(k+\alpha\sum_{j=max(0,i-\frac{n}{2})}^{min(C,i+\frac{n}{2})}{input(j,x,y)^2})^\beta$$ 


### 代码示例
```
# x是shape为[3,2]的张量

# out同样是shape[3,2]的张量，axis设置为1，表示将x中每个行向量做归一化
out = fluid.layers.l2_normalize(x, axis=1)
```
