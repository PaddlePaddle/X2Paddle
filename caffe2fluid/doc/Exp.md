## Exp


### [Exp](http://caffe.berkeleyvision.org/tutorial/layers/exp.html)
```
layer {
	name: "exp"
	type: "Exp"
	bottom: "data"
	top: "exp"	
	exp_param{
		base: -1
		scale: 1
		shift: 0
	}
}
```


### [paddle.fluid.layers.exp](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-186-exp)
```python
paddle.fluid.layers.exp(
	x,
	name = None
)
```  

### 功能差异
#### 计算机制的差异 
Caffe则有三个关于计算的参数，其计算公式为：  
$$
y=\begin{cases}
e^(shift+scale \times x),\quad x\leq 0 \\\\
base^(shift+scale \times x),\quad x>0
\end{cases}
$$
         

PaddlePaddle的计算公式为：$$y=e^x$$。 

