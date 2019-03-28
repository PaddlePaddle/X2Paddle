## Pooling

### [Pooling](http://caffe.berkeleyvision.org/tutorial/layers/pooling.html)
```
layer{
    name: "pool"
    type: "Pooling"
    bottom: "conv"
    top: "pool"
    pooling_param{
	pool: MAX
	kernel_size: 3	#必填项
	stride: 1
	pad: 0
    }
}
```
### [paddle.fluid.layers.pool2d](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-115-pool2d)
```python
paddle.fluid.layers.pool2d(
    input,
    pool_size,
    pool_type = 'max',
    pool_stride = 1,
    pool_padding = 0,
    global_pooling = False,
    use_cudnn = True,
    ceil_mode = False,
    name = None,
    exclusive = True
)
```  
  
### 功能差异
#### 计算输出高度和宽度的差异
计算池化的输出高度和宽度有两种方式，分别为向上取整（ceil）和向下取整（floor），其计算方式如下列所示：

**向上取整：**  
	`H_out = (H_in-ksize[0]+2*padding[0])/strides[0]+1`  
	`W_out = (W_in-ksize[1]+2*padding[1])/strides[1]+1`  

**向下取整：**  
	`H_out = (H_in-ksize[0]+2*padding[0]+strides[0]-1)/strides[0]+1`  
	`W_out = (W_in-ksize[1]+2*padding[1]+strides[1]-1)/strides[1]+1`    

Caffe：只能使用向上取整的方式来计算输入输出的大小。  
PaddlePaddle：可以使用`ceil_mode`参数来定义使用何种计算方式，当`ceil_mode=False`（默认值）时使用向下取整的方式来计算，反之为`True`时则使用向上取整的方式进行计算。  



#### 池化方式的差异
Caffe：提供了三种池化方式——最大池化、均值池化和随机池化（随机池化通过对像素点按照数值大小赋予概率，再按照概率进行亚采样）。  
PaddlePaddle：提供了两种池化方式——最大池化和均值池化。
 


#### 其他差异  
Caffe：无`exclusive`参数。  
PaddlePaddle：使用了一个`exclusive`参数，其代表在进行平均池化时是否忽略填充值。  


### 代码示例

```  
# Caffe示例：  
# 输入shape：(1,3,228,228)  
# 输出shape：(1,3,114,114)
layer{
    name: "pool"
    type: "Pooling"
    bottom: "conv"
    top: "pool"
    pooling_param{
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
pool1 = paddle.fluid.layers.pool2d(input = inputs , pool_size = 3, pool_type = 'max', pool_stride = 2, ceil_mode=False)
```  






