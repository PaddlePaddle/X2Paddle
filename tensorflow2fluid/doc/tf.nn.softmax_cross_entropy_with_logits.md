## tf.nn.softmax_cross_entropy_with_logits

### [tf.nn.rnn_cell.MultiRNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)

```python
tf.nn.softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
```

### [paddle.fluid.layers.softmax_with_cross_entropy](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#softmax-with-cross-entropy)
```python
paddle.fluid.layers.softmax_with_cross_entropy(
    logits, 
    label, 
    soft_label=False, 
    ignore_index=-100, 
    numeric_stable_mode=False, 
    return_softmax=False
)
```

### 功能差异

#### 标签类型
TensorFlow：`labels`只能使用软标签，其`shape`为`[batch, num_classes]`，表示样本在各个类别上的概率分布;  

PaddlePaddle：通过设置`soft_label`，可以选择软标签或者硬标签。当使用硬标签时，`label`的`shape`为`[batch, 1]`，`dtype`为`int64`；当使用软标签时，其`shape`为`[batch, num_classes]`，`dtype`为`int64`。

#### 返回值
TensorFlow：返回`batch`中各个样本的log loss；  

PaddlePaddle：当`return_softmax`为`False`时，返回`batch`中各个样本的log loss；当`return_softmax`为`True`时，再额外返回`logtis`的归一化值。


### 代码示例
```
# logits的shape为[32, 10], dtype为float32; label的shape为[32, 1], dtype为int64

# loss的shape为[32, 1], dtype为float32
loss = fluid.layers.softmax_with_cross_entropy(logits, label, soft_label=False)
                                               

```
