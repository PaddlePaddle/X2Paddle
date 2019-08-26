## 常见问题

**Q1. TensorFlow模型转换过程中，提示『Unknown shape for input tensor[tensor name: "input"]， Please define shape of input here』？**  
A：该提示信息表示无法从TensorFlow的pb模型中获取到输入tensor(tensor名为"input:)的shape信息，所以需要用户手动在提示后输入详细的shape信息，如None,224,224,3 其中None表示Batch 


**Q2. TensorFlow模型转换失败怎么解决?**  
A: 目前TensorFlow模型转换失败存在几个问题。1) 存在暂未支持的OP，此信息会在转换时输出; 2) NHWC优化导致部分参数出错；3）Batch维度带来的出错 4）其它

对于（1）问题，建议自行添加或发起Issue；

其中（2）、(3)、（4）问题目前没有明确的报错信息，当您遇到模型转换失败时，请尝试如下的步骤后，再进行转换测试

#### 关闭NHWC优化
TensorFlow的CV模型，大多的输入格式为`NHWC`，而Paddle目前仅支持`NCHW`，如若直接转换，需要在conv2d、pool2d等操作前后添加transpose解决，这样会带来性能的损耗。X2Paddle在模型转换过程中，对此问题进行了优化，避免transpose操作带来的性能问题，但目前仅在部分模型上进行了测试，不一定适用于其它模型，因此，如若模型转换存在问题时，我们建议你关闭NHWC的优化。

在模型转换时添加参数 --without_data_format_optimization
```
x2paddle -f tensorflow -m tf.pb -s pd-model --without_data_format_optimization
```

### 固定Batch大小
受限于不同框架的运行机制，在转换过程中，Batch维度也有一定可能会带来模型转换失败的问题。可以尝试固定Batch维度后再转换

在模型转换时添加参数 --define_input_shape
```
x2paddle -f tensorflow -m tf.pb -s pd-model --define_input_shape
```
如原tensorflow模型的输入shape为`[None, 224, 224, 3]`，可添加参数后，根据提示，把输入的shape修改为`[2, 224, 224, 3]`
