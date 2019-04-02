## Scale


### [Scale](http://caffe.berkeleyvision.org/tutorial/layers/scale.html)
```
layer {
    name: "scale"
    type: "Scale"
    bottom: "data"
    top: "scale"
    scale_param{
	axis: 1
	num_axes: 1
        filter{
	    type: "constant"
	    value: 2
	}
        bias_term: true
        bias_filter{
	    type: "constant"
	    value: 0.5
        }
    }
}
```


### [paddle.fluid.layers.scale](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-137-scale)
```python
paddle.fluid.layers.scale(
    x, 
    scale=1.0,  
    bias=0.0, 
    bias_after_scale=True, 
    act=None, 
    name=None
)
```  

### 功能差异
#### 输入参数
Caffe：参数`filter`和`bias_filter`在训练阶段会进行初始化，而在测试阶段`filter`和`bias_filter`的值都将被忽略。输入参数的维度由`axis`来定义，以大小为`100*3*40*60`的输入为例，其输入参数维度如下所示：  

|   axis值    | 可能维度1 | 可能维度2 | 可能维度3 |  可能维度4  |
| :---------: | :-------: | :-------: | :-------: | :---------: |
| axis==0 |    $$100$$    |   $$100\times3$$   | $$100\times3\times40$$  | $$100\times3\times40\times60$$ |
| axis==1 |     $$3$$     |   $$3\times40$$    |  $$3\times40\times60$$  |             |
| axis==2 |    $$40$$     |   $$40\times60$$   |           |             |
| axis==3 |    $$60$$     |           |           |             |

  
PaddlePaddle：不存在输入参数的维度问题，它的`scale`和`bias`在定义中设置了。  

#### 计算方式的差异
Caffe：只能在缩放之后添加bias。  
PaddlePaddle：可以通过设置`bias_after_scale`设置是在缩放之后还是之前添加bias。


#### 其他差异
Caffe：激活函数需要由另外一层完成。  
PaddlePaddle：可以通过设置`act`看是否在进行Scale后进行激活函数的操作。
