### tf.case

#### [tf.case](https://www.tensorflow.org/api_docs/python/tf/case)

```python
tf.case(
    pred_fn_pairs,
    default=None,
    exclusive=False,
    strict=False,
    name='case'
)
```

#### [paddle.fluid.layers.While](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#while)
```python
class paddle.fluid.layers.Switch(
	name=None
)
```

#### 功能差异

##### 使用方式
Tensorflow：用户采用定义`条件-函数对`的方式，创建一个`case`操作；

PaddlePaddle：用户通过在`switch`代码块中，定义`case`分支方式，实现`switch`操作。与tensorflow对比，在使用形式上更类似于传统的c/c++代码。


#### paddlepaddle代码示例
```
# 如下代码展示进行学习率的调度，当global_step超过某个数值时，学习率减小

# 定义学习率tensor
lr = fluid.layers.tensor.create_global_var(
    shape=[1],
    value=0.0,
    dtype='float32',
    persistable=True,
    name="learning_rate")
    
# 定义学习率常量
lr_0 = tensor.fill_constant(
    shape=[1], dtype='float32', value=1.0)
lr_1 = tensor.fill_constant(
    shape=[1], dtype='float32', value=0.1)

# 当global_step超过10000时，采用lr_1，否则采用lr_0
with fluid.layers.control_flow.Switch() as switch:
    with switch.case(global_step > 10000):
        fluid.layers.tensor.assign(input=lr_1, output=lr)
    with switch.default():
        fluid.layers.tensor.assign(input=lr_0, output=lr)

```