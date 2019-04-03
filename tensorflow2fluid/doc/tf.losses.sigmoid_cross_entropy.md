## tf.losses.sigmoid_cross_entropy

### [tf.losses.sigmoid_cross_entropy](https://www.tensorflow.org/api_docs/python/tf/losses/sigmoid_cross_entropy)

```python
tf.losses.sigmoid_cross_entropy(
    multi_class_labels,
    logits,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
```

### [paddle.fluid.layers.sigmoid_cross_entropy_with_logit](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#sigmoid_cross_entropy_with_logits)

```python
paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
    x, 
    label, 
    name=None
)
```

### 功能差异

#### 返回值类型

Tensorflow：通过控制`reduction`参数，返回结果可以是rank为0的tensor，也可以是shape与`logits`相同的tensor;  
PaddlePaddle：固定返回shape与`x`相同的tensor，表示每个样本在每个标签上的损失。

#### 其他

Tensorflow：通过`weights`，可以设置不同样本、不同label的权重；通过`label_smoothing`，可以控制对label进行平滑；  
PaddlePaddle：不支持调权与平滑功能。

### 代码示例
```
# x与label均是shape为[3,5]的tensor，表示三个样本，每个样本有5个类别

# out是shape为[3,5]的tensor，表示每个样本在每个类别上的loss
out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)


```
