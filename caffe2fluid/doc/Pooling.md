## Pooling

### [Pooling](http://caffe.berkeleyvision.org/tutorial/layers/pooling.html)
```
layer{
    name: "pool"
    type: "Pooling"
    bottom: "data"
    top: "pool"
    pooling_param {
	pool: MAX
	kernel_size: 3    # 必填项
	stride: 1
	pad: 0
    }
}
```
### [paddle.fluid.layers.pool2d](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#permalink-119-pool2d)
```python
paddle.fluid.layers.pool2d(
    input,
    pool_size,
    pool_type='max',
    pool_stride=1,
    pool_padding=0,
    global_pooling=False,
    use_cudnn=True,
    ceil_mode=False,
    name=None,
    exclusive=True
)
```  
  
### 功能差异
#### 输出大小 
Caffe：输出大小计算方式如下所示，
```
H_out = (H_in-ksize[0]+2*padding[0])/strides[0]+1
W_out = (W_in-ksize[1]+2*padding[1])/strides[1]+1
```

PaddlePaddle：`ceil_mode`为`Ture`时，输出大小计算方式与Caffe一致；当`ceil_mode`为`False`时，输出大小计算方式如下所示，
```
# ceil_model为False时，计算公式
H_out = (H_in-ksize[0]+2*padding[0]+strides[0]-1)/strides[0]+1
W_out = (W_in-ksize[1]+2*padding[1]+strides[1]-1)/strides[1]+1
```

#### 池化方式
Caffe：通过`pool`参数设置，支持`MAX`, `AVE`和`STOCHASTIC`三种池化方式；  
PaddlePaddle：通过`pool_type`参数设置，支持`max`和`avg`两种池化方式。

#### 其他 
Caffe：无`exclusive`参数；  
PaddlePaddle：`exclusive`参数为`True`的情况下，`avg`平均池化过程中会忽略填充值。


### 代码示例

```  
# Caffe示例：  
# 输入shape：(1,3,228,228)  
# 输出shape：(1,3,114,114)
layer{
    name: "pool"
    type: "Pooling"
    bottom: "data"
    top: "pool"
    pooling_param {
	pool: MAX
	kernel_size: 3	
	stride: 2
    }
}
```  
``` python
# PaddlePaddle示例：  
# 输入shape：(1,3,228,228)  
# 输出shape：(1,3,113,113)
pool1 = paddle.fluid.layers.pool2d(input = inputs , pool_size = 3, 
                                   pool_type = 'max', pool_stride = 2, 
				   ceil_mode=False)
```  






