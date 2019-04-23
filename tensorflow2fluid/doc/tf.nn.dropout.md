
## tf.dropout

### [tf.nn.dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout)
``` python
tf.nn.dropout(
    x,
    keep_prob=None,
    noise_shape=None,
    seed=None,
    name=None,
    rate=None
)
```

### [paddle.fluid.layers.dropout](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#cn-api-fluid-layers-dropout)
``` python
paddle.fluid.layers.dropout(
    x, 
    dropout_prob, 
    is_test=False, 
    seed=None, 
    name=None, 
    dropout_implementation='downgrade_in_infer'
)
```

### 功能差异
#### 丢弃概率
TensorFlow：使用`keep_prob`表示保留单元输出的概率，等价于`1-rate`；  
PaddlePaddle：使用`dropout_prob`表示将单元输出设置为0的概率，即其丢弃概率；

#### dropout独立性
TensorFlow：通过设置一个可以广播到x的`noise_shape`，可以控制dropout的独立性；  
PaddlePaddle：暂无此设置。

#### 实现方式
TensorFlow：在训练时，被保留的单元输出要乘上`1/keep_prob`的系数，而在测试时，直接关闭dropout。
PaddlePaddle：通过设置`dropout_implementation`有不同的实现。当设置为`downgrade_in_infer`时，在训练时，保留单元直接被输出，而测试时所有单元乘以`1-dropout_prob`的系数；当设置为`upscale_in_train`时，则与tensorflow的实现一致。

### 代码示例
```python
# 输入 tensor t 为[[1,2],[3,4]]

# 第0维前面padding长度为0，后面padding长度为1；第1维前面padding长度为1，后面padding长度为2
out = fluid.layers.dropout(t, dropout_prob=0.2, dropout_implementation="upscale_in_train")

# inference 时关闭dropout
inference_program = fluid.default_main_program().clone(for_test=True)
```
