## ArgMax


### [ArgMax](http://caffe.berkeleyvision.org/tutorial/layers/argmax.html)
```
layer {
	name: "argmax"
	type: "ArgMax"
	bottom: "data"
	top: "argmax"	
	argmax_param{
		out_max_val: false
		top_k: 1
		axis: 0
	}
}
```


### [paddle.fluid.layers.argmax](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-204-argmax)
```
paddle.fluid.layers.argmax(
	x,
	axis = 0
)
```  

### 功能差异
#### 输出的差异
Caffe：可以通过设置设置`top_k`使输出为前k大的索引，同时可以设置`out_max_val`为true来使输出为前k大的数值。                                    
PaddlePaddle：只能输出最大值的索引。
