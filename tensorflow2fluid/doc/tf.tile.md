
## tf.tile

### [tf.tile](https://www.tensorflow.org/api_docs/python/tf/constant)
``` python
tf.tile(
    input,
    multiples,
    name=None
)
```

### [paddle.fluid.layers.expand](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#cn-api-fluid-layers-expand)
``` python
paddle.fluid.layers.expand(
    x, 
    expand_times, 
    name=None)
```

### 功能差异：
#### 参数类型差异：
>  tensorflow：value可以是python list，也可以是variable。
>  paddlepaddle：value必须是python list。

## paddlepaddle示例:
```python
# 输入 tensor t 为[[1,2],[3,4]]

# 当expand_times 为[1, 2]时，输出 tensor out 为[[1,2,1,2],[3,4,3,4]]
out = fluid.layers.expand(t, [1,2])  
