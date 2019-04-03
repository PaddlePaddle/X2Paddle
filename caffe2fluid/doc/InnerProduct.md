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
#### 参数初始化
Caffe：Layer定义中共有两个结构体`param`用于设置局部学习率和权值衰减因子，其中第一个用于设置卷积核，第二个则用于设置偏值项；卷积核和偏置项的初始化参数在`convolution_param`中进行设置；是否使用偏置项可以使用`bias_term`进行设置。  
PaddlePaddle：卷积核和偏置项的参数分别使用`param_attr`和`bias_attr`进行配置，配置参数如下所示，此外将`bias_attr`直接设为`False`表示不使用偏置项。
```
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
#### 参数格式
Caffe：输入参数的数据格式是`(filter_num, channel*height*width)`。  
PaddlePaddle：在`num_flatten_dims=1`且输入数据维度为4的情况下，其输入参数的数据格式是`(channel*height*width, filter_num)`；其他情况下输入参数的数据格式不一定是二维的，但是`filter_num`都始终放在第二维。
#### 输入数据扁平化
Caffe：将输入数据的第一维默认为batch size，其他剩余的几个维度扁平化压缩成一个向量进行全连接的计算。                     
PaddlePaddle：通过设置`num_flatten_dims`的值，使`输入数据的维度-num_flatten_dim`个维度扁平化压缩成一个向量进行全连接计算。


#### 其他差异
Caffe：需要在另一个层中定义激活函数。  
PaddlePaddle：可以通过设置`act`这一参数来确定输出的激活函数。
