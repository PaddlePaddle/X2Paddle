## Deconvolution


### [Deconvolution](http://caffe.berkeleyvision.org/tutorial/layers/deconvolution.html)
```
layer {
    name: "deconv"
    type: "Deconvolution"
    bottom: "data"
    top: "deconv"
    # 卷积核的局部学习率和权值衰减因子
    param {
	lr_mult: 1
	decay_mult: 1
    }
    # 偏置项的局部学习率和权值衰减因子
    param {
	lr_mult: 2
	decay_mult: 0
    }
    convolution_param {
	num_output: 20    # 必填项
	kernel_size: 3    # 必填项
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


### [paddle.fluid.layers.conv2d_transpose](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#permalink-46-conv2d_transpose)
```python
paddle.fluid.layers.conv2d_transpose(
    input,
    num_filters,
    output_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=None,
    param_attr=None,
    bias_attr=None,
    use_cudnn=True,
    act=None,
    name=None
)
```  

### 功能差异
#### 参数初始化
Caffe：Layer定义中共有两个结构体`param`用于设置局部学习率和权值衰减因子，其中第一个用于设置卷积核，第二个则用于设置偏值项；卷积核和偏置项的初始化参数在`convolution_param`中进行设置；是否使用偏置项可以使用`bias_term`进行设置；  
PaddlePaddle：卷积核和偏置项的参数分别使用`param_attr`和`bias_attr`进行配置，配置参数如下所示，此外将`bias_attr`直接设为`False`表示不使用偏置项。
```python
paddle.fluid.ParamAttr(
    name=None, 
    initializer=None, 
    learning_rate=1.0, 
    regularizer=None, 
    trainable=True, 
    gradient_clip=None, 
    do_model_average=False
)
```
#### 空洞卷积
Caffe：无法使用空洞卷积；                  
PaddlePaddle：使用`dilation`参数来设置空洞卷积。
