## LRN


### [LRN](http://caffe.berkeleyvision.org/tutorial/layers/lrn.html)
```
layer {
    name: "lrn"
    type: "LRN"
    bottom: "data"
    top: "lrn"	
    lrn_param {
	local_size: 5
	alpha: 1
	beta: 5
	norm_region: "ACROSS_CHANNELS"
    }
}
```


### [paddle.fluid.layers.lrn](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-99-lrn)
```python
paddle.fluid.layers.lrn(
    input, 
    n=5, 
    k=1.0, 
    alpha=0.0001, 
    beta=0.75, 
    name=None
)
```  

### 功能差异
#### 参数差异
Caffe：参数`norm_region`支持`ACROSS_CHANNELS`和`WITHIN_CHANNEL`两种模式；  
PaddlePaddle：默认且仅支持`ACROSS_CHANNELS`模式。

#### 计算机制
Caffe：在`ACROSS_CHANNELS`模式下，计算公式如下， 
$$output(i,x,y)=input(i,x,y)/(1+\frac{\alpha}{n}\sum_{j=max(0,i-\frac{n}{2})}^{min(C,i+\frac{n}{2})}{input(j,x,y)^2})^\beta$$  
位移值固定为1，计算缩放参数的同时除以通道数目  

PaddlePaddle：计算公式如下，
$$output(i,x,y)=input(i,x,y)/(k+\alpha\sum_{j=max(0,i-\frac{n}{2})}^{min(C,i+\frac{n}{2})}{input(j,x,y)^2})^\beta$$  
能通过设置k来定义位移数。

