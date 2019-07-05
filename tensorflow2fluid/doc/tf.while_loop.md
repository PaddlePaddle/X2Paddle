## tf.while_loop

### [tf.while_loop](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/while_loop)

```python
tf.while_loop(
    cond,
    body,
    loop_vars,
    shape_invariants=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    name=None,
    maximum_iterations=None,
    return_same_structure=False
)
```

### [paddle.fluid.layers.While](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#while)
```python
paddle.fluid.layers.While(
    cond, 
    is_test=False, 
    name=None
)
```

### 功能差异

#### 使用方式
TensorFlow：用户通过函数的方式定义`cond`和`body`，在循环体中操纵的是循环变量`loop_vars`，返回值为`tensor`或其`list`;  
PaddlePaddle：用户通过op的方式定义`cond`，然后在`block`中实现循环体。*注意，在循环体中用户需要更新`cond`，具体可参数代码示例*。

#### 其他
TensorFlow：支持设置最大迭代次数`maximum_iterations`及并行迭代`parallel_iterations`;  
PaddlePaddle：不涉及最大迭代次数及并行。


### 代码示例
```
# 如下代码片段实现从0到5循环叠加i
i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
limit = fluid.layers.fill_constant(shape=[1], dtype='int64', value=5)

# 定义条件
cond = layers.less_than(x=i, y=limit)
while_op = layers.While(cond=cond)

# 定义循环体
with while_op.block():
	 # 更新i
    i = layers.increment(x=i, in_place=True)
    # 更新条件状态
    layers.less_than(x=i, y=limit, cond=cond)
```