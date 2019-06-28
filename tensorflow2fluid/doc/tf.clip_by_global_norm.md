## tf.clip_by_global_norm

### [tf.clip_by_global_norm](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/clip_by_global_norm)

```python
tf.clip_by_global_norm(
    t_list,
    clip_norm,
    use_norm=None,
    name=None
)
```

### [paddle.fluid.clip.GradientClipByGlobalNorm](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/clip_cn.html#gradientclipbyglobalnorm)

```python
paddle.fluid.clip.GradientClipByGlobalNorm(
    clip_norm, 
    group_name='default_group'
)
```

### 功能差异

#### 使用方式

TensorFlow：采用函数调用形式，输入需要执行global_norm裁剪的tensor，返回裁剪后的结果；  

PaddlePaddle：采用类对象定义形式，使用`set_gradient_clip`函数设置`GradientClipByGlobalNorm`对象为裁剪方式。

#### 其他
TensorFlow：使用`use_norm`支持外部设置global_norm，若没有设置则从`t_list`计算得到；  

PaddlePaddle：不支持外部设置。

### 代码示例
```
# 获取待裁剪的tensor列表
p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

with fluid.program_guard(main_program=prog_clip):
    # 设置裁剪方式
    fluid.clip.set_gradient_clip(
      fluid.clip.GradientClipByGlobalNorm(clip_norm=2.0))
    
    # 执行裁剪并获取结果
    p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)

```