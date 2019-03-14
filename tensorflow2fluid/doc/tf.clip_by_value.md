
### tf.clip_by_value

#### [tf.clip_by_value](https://www.tensorflow.org/api_docs/python/tf/clip_by_value)
``` python
tf.clip_by_value(
    t,
    clip_value_min,
    clip_value_max,
    name=None
)
```

#### [paddle.fluid.layers.clip](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#cn-api-fluid-layers-clip)
``` python
paddle.fluid.layers.clip(
    x, 
    min, 
    max, 
    name=None
)
```

#### 功能差异
##### 参数类型
tensorflow：`clip_value_min`/`clip_value_max`可以是python scalar，也可以是variable；  
paddlepaddle：`min`/`max`必须是python scalar。

#### paddlepaddle代码示例
```python
# 输入 tensor t 为[[-1,2],[3,-4]]

# 输出 tensor out 为[[0,1],[1,0]]
out = fluid.layers.clip(t, 0, 1)  
