### tf.reverse_sequence

#### [tf.reverse_sequence](https://www.tensorflow.org/api_docs/python/tf/reverse_sequence)

```python
tf.reverse_sequence(
    input,
    seq_lengths,
    seq_axis=None,
    batch_axis=None,
    name=None,
    seq_dim=None,
    batch_dim=None
)
```

#### [paddle.fluid.layers.sequence_reverse](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#sequence_reverse)

```python
paddle.fluid.layers.sequence_reverse(
    x, 
    name=None
)
```

#### 功能差异

##### 输入tensor类型

Tensorflow：`reverse_sequence`中，`input`是一个带padding的tensor，每个序列都会被填充到相同长度;  
PaddlePaddle：`sequence_reverse`中，`x`是一个[LoDTensor](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/fluid_cn.html#lodtensor)，
不需要进行填充；

##### 输入参数类型

Tensorflow：通过`seq_axis`和`batch_axis`指定序列维度与batch维度；同时使用`seq_lengths`来表示每个序列的长度，属于序列的部分会被翻转，padding部分则被
保留；  
PaddlePaddle：由于`LoDTensor`本身已经携带序列信息，因而不需要用户提供除了输入tensor外的额外参数；

#### paddlepaddle代码示例
```
# x是shape为[5, 6]的LoDTensor，其LoD信息为{0, 2, 5}，表示两个序列，长度分别是2和3

# out同样也是shape为[5, 6]的LoDTensor，LoD信息为{0, 2, 5}，表示两个序列，分别是x中两个序列反转后的结果
# out[0:2, 6] = x[2:0:-1, 6]
# out[2:5, 6] = x[5:2:-1, 6]
out = fluid.layers.sequence_reverse(x)
```
