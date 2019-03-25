## ReLU


### [ReLU](http://caffe.berkeleyvision.org/tutorial/layers/relu.html)
```
layer {
	name: "relu"
	type: "ReLU"
	bottom: "data"
	top: "relu"
	relu_param{
		negative_slope: 0
	}	
}
```


### [paddle.fluid.layers.relu](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-128-relu)
```python
paddle.fluid.layers.relu(
	x, 
	name=None
)
```
和  
```
paddle.fluid.layers.leaky_relu(
	x, 
	alpha = 0.02,
	name=None
)
```


### 功能差异
#### 实现的差异
Caffe：使用这个Layer即可分别实现ReLU和Leaky ReLU两个功能。     
$$
y=\begin{cases}
x,\quad x\geq 0 \\\\
\alpha \times x,\quad x<0
\end{cases}
$$       
PaddlePaddle：只能通过两个函数分别实现ReLU和Leaky ReLU。         


$$
y=max(x,\alpha \times x)
$$
