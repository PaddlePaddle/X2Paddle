## tf.stop_gradient

### [tf.stop_gradient](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/stop_gradient)
``` python
tf.stop_gradient(
    input,
    name=None
)
```

### PaddlePaddle实现
TensorFlow中，使用`stop_gradient`表示该tensor不需要进行bp。而在PaddlePaddle中，每个tensor具有`stop_gradient`的属性，用户可以将该属性直接设置成`True`/`False`。

## 代码示例
```python
# 将tensor t设置成不需要bp
t.stop_gradient = True