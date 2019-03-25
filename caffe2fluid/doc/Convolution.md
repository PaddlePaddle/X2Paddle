## Convolution


### [Convolution](http://caffe.berkeleyvision.org/tutorial/layers/convolution.html)
```
layer{
	name: "conv"
	type: "Convolution"
	bottom: "data"
	top: "conv"
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
	convolution_param{
		num_output: 20	#必填项
		kernel_size: 5	#必填项
		stride: 1
		pad: 0
		group: 1
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


### [paddle.fluid.layers.conv2d](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-45-conv2d)
```
paddle.fluid.layers.conv2d(
	input,
	num_filters,
	output_size,
	stride = 1,
	padding = 0,
	dilation = 1,
	groups = None,
	param_attr=None,
	bias_attr=None,
	use_cudnn=True,
	act=None,
	name=None
)
```  

### 功能差异
#### 参数初始化的差异
Caffe：第一个`param`负责设置卷积核的局部学习率和权值衰减因子，第二个`param`则负责设置偏置项的局部学习率和权值衰减因子；而卷积核和偏置项的在`convolution_param`中进行设置；是否使用偏置项可以使用`bias_term`进行设置。           
PaddlePaddle：卷积核和偏置项的多处设置均分别在一个参数——`param_attr`/`bias_attr`中完成所有操作。二者的默认值为None，而ParamAttr是一个初始化结果，其可以通过`paddle.fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)`获得；bias_attr同时可以是设置为布尔型，用来表示是否使用偏置项。
#### 空洞卷积的使用
Caffe：无法使用空洞卷积。                  
PaddlePaddle：使用`dilation`来设置空洞卷积。
