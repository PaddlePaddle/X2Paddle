## Power


### [Power](http://caffe.berkeleyvision.org/tutorial/layers/power.html)
```
layer {
    name: "power"
    type: "Power"
    bottom: "data"
    top: "power"	
    power_param{
	power: 1
	scale: 1
	shift: 0
    }
}
```


### [paddle.fluid.layers.power](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-117-pow)
```python
paddle.fluid.layers.power(
    x,
    factor = 1.0,
    name = None
)
```  

### 功能差异
#### 计算机制的差异
Caffe：有三个关于计算的参数，其计算公式为：$$y=(shift+scale \times x)^2$$            
PaddlePaddle：只有一个关于计算的参数`factor`，其计算公式为：$$y=x^factor$$
