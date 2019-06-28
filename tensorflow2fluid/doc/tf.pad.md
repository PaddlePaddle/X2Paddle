## tf.pad

### [tf.pad](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/pad)
``` python
tf.pad(
    tensor,
    paddings,
    mode='CONSTANT',
    name=None,
    constant_values=0
)
```

### [paddle.fluid.layers.pad](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#cn-api-fluid-layers-pad)
``` python
paddle.fluid.layers.pad(
    x, 
    paddings, 
    pad_value=0.0, 
    name=None
)
```

### 功能差异
#### padding方式
TensorFlow：支持采用三种模式进行padding，不同padding模式决定pad的值是什么，包括constant、symmetric和reflect。padding的shape为(rank, 2)，表示每一维前后padding的长度  

PaddlePaddle：目前仅支持采用常量进行padding；指定padding长度时，采用一个一维列表表示，其长度为输入rank的两倍，连续的两个值表示某维度上前、后进行padding的长度

### 代码示例
```python
# 输入 tensor t 为[[1,2],[3,4]]

# 第0维前面padding长度为0，后面padding长度为1；第1维前面padding长度为1，后面padding长度为2
out = fluid.layers.pad(t, paddings=[0,1,1,2])  
```