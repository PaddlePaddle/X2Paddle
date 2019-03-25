## InnerProduct
### [InnerProduct](http://caffe.berkeleyvision.org/tutorial/layers/innerproduct.html)
```
layer{
	name: "fc"
	type: "InnerProduct"
	bottom: "data"
	top: "fc"
	#卷积核的局部学习率和权值衰减因子
	param{
		lr_mult: 1
		decay_mult: 1
	}
	#偏置项的局部学习率和权值衰减因子
	param{
		lr_mult: 2
		decay_mult: 0
	}
	InnerProduct{
		num_output: 20	#必填项
		bias_term: True
		weight_filler {
			type: "gaussian"
			value: 0.01
		}
		bias_filler {
			type: "constant"
			value: 0
		}
	}
}
```


### [paddle.fluid.layers.fc](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-71-fc)
```python
paddle.fluid.layers.fc(
	input,
	size,
	num_flatten_dims=1,
	param_attr=None,
	bias_attr=None,
	act=None,
	is_test=False,
	name=None
)
```  

### 功能差异
#### 参数初始化的差异

Caffe：第一个`param`负责设置卷积核的局部学习率和权值衰减因子，第二个`param`则负责设置偏置项的局部学习率和权值衰减因子；而卷积核和偏置项的在`convolution_param`中进行设置；是否使用偏置项可以使用`bias_term`进行设置。  
PaddlePaddle：Caffe中的卷积核和偏置项的多处设置均分别在一个参数——`param_attr`/`bias_attr`中完成所有操作。二者的默认值为None，而ParamAttr是一个初始化结果，其可以通过`paddle.fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)`获得；bias_attr同时可以是设置为布尔型，用来表示是否使用偏置项。
#### 参数格式的差异
Caffe：输入参数的数据格式是`(filter_num, channel*height*width)`。  
PaddlePaddle：在`num_flatten_dims=1`且维度为4的情况下，其输入参数的输入数据格式则是`(channel*height*width, filter_num)`，而其他不管什么情况PaddlePaddle的filter_num都始终应该放在第二维。
#### 输入数据扁平化的差异
Caffe：将输入数据的第一维默认为batch size，其他剩余的几个维度扁平化压缩成一个向量进行全连接的计算。                     
PaddlePaddle：通过设置`num_flatten_dims`的值，确认后`rank(input)-num_flatten_dim`个维度扁平化压缩成一个向量进行全连接计算。


#### 其他差异
Caffe：需要在另一个层中定义激活函数。  
PaddlePaddle：可以通过设置`act`这一参数来确定输出的激活函数。
