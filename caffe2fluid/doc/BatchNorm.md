## BatchNorm


### [BatchNorm](http://caffe.berkeleyvision.org/tutorial/layers/batchnorm.html)
```
layer {
    name: "bn"
    type: "BatchNorm"
    bottom: "data"
    top: "bn"
    batch_norm_param{
        use_global_stats: true
    	moving_average_fraction: 0.999
    	eps: 0.00001
    }
}
```


### [paddle.fluid.layers.batch_norm](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-36-batch_norm)
```python
paddle.fluid.layers.batch_norm(
    input, 
    act=None, 
    is_test=False, 
    momentum=0.9, 
    epsilon=1e-05, 
    param_attr=None, 
    bias_attr=None, 
    data_layout='NCHW', 
    in_place=False, 
    name=None, 
    moving_mean_name=None, 
    moving_variance_name=None, 
    do_model_average_for_mean_and_var=False, 
    fuse_with_relu=False, 
    use_global_stats=False
)
```  

### 功能差异
#### 输出结果的差异
Caffe：输出只是单纯的使用均值和方差进行归一化计算，没有缩放变换这一过程，若要完成后面这一过程需要搭配Scale层进行使用。  
PaddlePaddle：完成了归一化和缩放变换两个过程，是完整的一个Batch Normalization过程。


#### 输入参数的差异
Caffe：共需要3个输入参数：均值向量、方差向量、滑动系数。    
PaddlePaddle：共需要4个输入参数：均值向量、方差向量、缩放变量`param_attr`和缩放变量`bias_attr`。
#### 计算方式的差异
Caffe：输入的均值和方差都需要与滑动系数进行计算得出新均值和方差，再进行归一化处理。    
PaddlePaddle：直接使用输入的均值和方差进行归一化处理。


#### 其他差异
Caffe：激活函数需要由另外一层完成。  
PaddlePaddle：可以通过设置`act`和`fuse_with_relu`看是否在进行Batch Normalization后进行激活函数的操作。
