## SofmaxWithLoss


### [SofmaxWithLoss](http://caffe.berkeleyvision.org/tutorial/layers/softmaxwithloss.html)
```
layer {
    name: "loss"
    type: "SoftmaxWithLoss"
    bottom: "pred"
    bottom: "label"
    top: "loss"
    loss_param{
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
    soft_label = False,
    ignore_index = -100,
    numeric_stable_mode = False, 
    return_softmax = False
)
```  

### 功能差异
#### 计算机制

Caffe：只可以使用硬标签的输入，同时进行预处理操作。                     
PaddlePaddle：可以使用`soft_label`来设置是使用软标签（True）还是硬标签（False）；将`numeric_stable_mode`设为True，同时在GPU环境下运行，可是在使用硬标签之前先进行预处理。此外，软标签和硬标签的label输入略有不同，当log概率的输入大小为`N*K`时（`N`代表batch size，`K`代表类别数量），软标签的输入大小为`N*K`，其重的数值数据类型为`float`或者`double`，每一个batch中的值都是0或者1（1代表属于这个类别，0则代表不属于）；硬标签的输入大小为`N*1`，其重的数值数据类型为`int`，每一个batch中的值都是大于等于0且小于K（代表属于某一个类别）。在Caffe中，则只可以使用硬标签的输入，同时进行预处理操作。 

> 计算softmax的loss时，根据每个样本是否被分配至多个类别中可以分为两类——硬标签和软标签，具体如下：  
  
> **硬标签：** 即one-hot label，每个样本仅分到一个类别中。在硬标签中，根据是否对未初始化的log概率进行预处理，又可以分为两类，预处理主要是完成对每个样本中的每个log概率减去该样本中的最大的log概率。  
 
> **软标签：** 每个样本至少被分配到一个类别中。  
 
#### 输出结果
Caffe：输出是对所有样本的loss进行归一化后的结果，同时根据`normalize`和`normalization`的设置，归一化形式略有不同，当`normalization`是FULL或0时整个loss取和后除以batch的大小，当`normalization`是VALID或1时整个loss取和后除以除`ignore_label`以外的样本数，为NONE时则取和；当`normalization`未设置时，采用`normalize`的值进行判断，若`normalize==1`则归一化方式是VALID，若`normalize==0`则归一化方式是FULL。                    
PaddlePaddle：输出是每个样本的loss所组成的一个向量，同时如果将参数`return_softmax`设为True，则输出的是loss向量和softmax值组成的一个元组。

### 代码示例
```  
# Caffe示例：
# pred输入shape：(100,10)  
# label输入shape：(100,1)  
# 输出shape：()
layer {
    name: "loss"
    type: "SoftmaxWithLoss"
    bottom: "pred"
    bottom: "label"
    top: "loss"
    loss_param{
	ignore_label: -1
	normalize: 0
	normalization: FULL

    }
}
```

  
```python  
# PaddlePaddle示例：
# pred输入shape：(100,10)  
# label输入shape：(100,1)  
# 输出shape：(10,1)
softmaxwithloss= paddle.fluid.layers.softmax_with_cross_entropy(logits = logs, label = labels, soft_label=False, ignore_index=-100, numeric_stable_mode=False, return_softmax=False)
```
