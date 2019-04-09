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
#### 计算机制
Caffe：`BatchNorm`仅做了归一化计算，需结合`Scale`层进行缩放变换；  
PaddlePaddle：包括归一化计算和缩放变换，`param_attr`和`bias_attr`即为缩放变换的设置参数。
