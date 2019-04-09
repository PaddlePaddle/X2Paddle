## ArgMax


### [ArgMax](http://caffe.berkeleyvision.org/tutorial/layers/argmax.html)
```
layer {
    name: "argmax"
    type: "ArgMax"
    bottom: "data"
    top: "argmax"	
    argmax_param {
	out_max_val: false
	top_k: 1
	axis: 0
    }
}
```


### [paddle.fluid.layers.argmax](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-204-argmax)
```python
paddle.fluid.layers.argmax(
    x,
    axis=0
)
```  

### 功能差异
#### 计算机制
Caffe：可通过`top_k`和`out_max_val`参数设置得到前`k`的索引或数值；                            
PaddlePaddle：只能输出最大值的索引；
