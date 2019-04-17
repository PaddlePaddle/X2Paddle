## SofmaxWithLoss


### [SofmaxWithLoss](http://caffe.berkeleyvision.org/tutorial/layers/softmaxwithloss.html)
```
layer {
    name: "loss"
    type: "SoftmaxWithLoss"
    bottom: "logits"
    bottom: "label"
    top: "loss"
    softmax_param {
        axis: 1
    }
    loss_param {
	ignore_label: -1
	normalize: 0
	normalization: FULL
    }
}
```


### [paddle.fluid.layers.softmax_with_cross_entropy](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-164-softmax_with_cross_entropy)
```python
paddle.fluid.layers.softmax_with_cross_entropy(
    logits,
    label,
    soft_label=False,
    ignore_index=-100,
    numeric_stable_mode=True, 
    return_softmax=False
)
```  

### 功能差异
#### 输入数据
Caffe：输入数据（`x`）的维度最大是4维(`N*C*H*W`)；                 
PaddlePaddle：输入数据(`x`和`label`)的维度只能是2维（`N*K`）。
#### 输入格式
Caffe: 采用硬标签方式输入，同时进行预处理操作(为了避免上溢出和下溢出，对输入的每个值减去batch中该位置上的最大值);  
PaddlePaddle：通过参数`soft_label`的设定，支持硬标签和软标签两种输入。  
> 计算softmax的loss时，根据每个样本是否被分配至多个类别中可以分为两类——硬标签和软标签  
> **硬标签：** 即one-hot label，每个样本仅分到一个类别中。在硬标签中，根据是否对未初始化的log概率进行预处理，又可以分为两类，预处理主要是完成对每个样本中的每个log概率减去该样本中的最大的log概率  
> **软标签：** 每个样本至少被分配到一个类别中
 
#### 输出结果
Caffe：输出是对所有样本的loss进行归一化后的结果，归一化的方式由`normalization`和`normalize`参数决定；
```
归一化形式:
1. 当`normalization`是FULL或0时，整个loss取和后除以batch的大小.
2. 当`normalization`是VALID或1时，整个loss取和后除以除`ignore_label`以外的样本数。
3. 当`normalization`是NONE时，则loss取和.
4. 当`normalization`未设置时，采用`normalize`的值进行判断，若`normalize==1`则归一化方式是VALID，若`normalize==0`则归一化方式是FULL。
```
PaddlePaddle：输出是每个样本的loss所组成的一个向量，同时如果将参数`return_softmax`设为True，则输出的是loss向量和softmax值组成的一个元组。

### 代码示例
```  
# Caffe示例：
# logits输入shape：(100,10)  
# label输入shape：(100,1)  
# 输出shape：()
layer {
    name: "loss"
    type: "SoftmaxWithLoss"
    bottom: "logits"
    bottom: "label"
    top: "loss"
    loss_param {
	ignore_label: -1
	normalize: 0
	normalization: FULL

    }
}
```

  
```python  
# PaddlePaddle示例：
# logits输入shape：(100,10)  
# label输入shape：(100,1)  
# 输出shape：(10,1)
softmaxwithloss = fluid.layers.softmax_with_cross_entropy(logits=logs, label=labels, 
							soft_label=False, ignore_index=-100, 
							numeric_stable_mode=True, 
							return_softmax=False)
```
