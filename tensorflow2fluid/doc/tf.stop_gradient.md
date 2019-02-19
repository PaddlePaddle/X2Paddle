
## tf.stop_gradient

### [tf.stop_gradient](https://www.tensorflow.org/api_docs/python/tf/stop_gradient)
``` python
tf.stop_gradient(
    input,
    name=None
)
```

tensorflow中，使用stop_gradient表示该tensor不需要进行bp。而在paddlepaddle中，每个tensor具有stop_gradient的属性，用户可以将该属性直接设置成True/False。

## paddlepaddle示例:
```python
# 将tensor t设置成不需要bp
t.stop_gradient = True
