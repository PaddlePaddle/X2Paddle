
### tf.one_hot

#### [tf.one_hot](https://www.tensorflow.org/api_docs/python/tf/one_hot)
``` python
tf.one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
```

#### [paddle.fluid.layers.one_hot](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#one-hot)
``` python
layers.one_hot(
    input, 
    depth
)
```

#### 功能差异：
tensorflow：indices shape 没有限定；支持设置on与off的值；支持使用axis设置depth维所处位置。

paddlepaddle：input 必须是二维tensor，shape为(batch, 1)；depth必须是python int标量；限定on与off value为1和0。

#### paddlepaddle示例:
```python
# 输入 tensor t 为[[1],[2]]

# depth 为3时，输出 tensor out 为[[0, 1, 0], [0, 0, 1]]
out = fluid.layers.one_hot(t, 3)
```
