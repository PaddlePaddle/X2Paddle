## tf.slice

### [tf.slice](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/slice)
``` python
tf.slice(
    input_,
    begin,
    size,
    name=None
)
```

### [paddle.fluid.layers.slice](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#cn-api-fluid-layers-slice)
``` python
paddle.fluid.layers.slice(
    input, 
    axes, 
    starts, 
    ends
)
```

### 功能差异
#### 参数类型
TensorFlow：`begin/size`可以是python list，也可以是变量类型；  
PaddlePaddle：`axes/starts/ends`只能是python list。

#### 参数种类
TensorFlow：使用`begin`指定要开始截取tensor的位置，使用`size`指定截取长度，必须描述所有的轴；  
PaddlePaddle：采用`axes`指定要操作的轴，未指定的轴默认全部截取，使用`starts`、`ends`分别指定截取tensor的开始与结束位置，注意采用的是先闭后开[start, end)的写法。


### 代码示例
```python
# 输入 tensor t 为[[0,1,2,3],[4,5,6,7],[8,9,10,11]]

# 输出 tensor out 为[[1,2],[5,6]]
out = fluid.layers.slice(t, axes=[0,1], starts=[0,1], ends=[2,3])  

# 输出 tensor out 为[[1,2],[5,6],[9,10]]
out = fluid.layers.slice(t, axes=[1], starts=[1], ends=[3])
```