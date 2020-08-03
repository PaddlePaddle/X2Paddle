## 常见问题

**Q1. TensorFlow模型转换过程中，提示『Unknown shape for input tensor[tensor name: "input"]， Please define shape of input here』？**  
A：该提示信息表示无法从TensorFlow的pb模型中获取到输入tensor(tensor名为"input:)的shape信息，所以需要用户手动在提示后输入详细的shape信息，如None,224,224,3 其中None表示Batch


**Q2. TensorFlow模型转换失败怎么解决?**  
A: 如果并非是由缺少OP导致，那可能是由于TensorFlow模型转换时（NHWC->NCHW格式转换导致），在这种情况下，采用如下方式进行转换，同时固化输入大小的方式，继续尝试转换，见如下命令，转换过程中，根据提示，输入相应tensor的固化shape大小
```
x2paddle -f tensorflow -m tf.pb -s pd-model --without_data_format_optimization --define_input_shape
```

> 1. 目前Tensorflow的CV模型大部分均为`NHWC`的输入格式，而Paddle的默认输入格式为`NCHW`，因此X2Paddle在转换过程中，会对如`axis`， `shape`等参数进行转换，适应Paddle的NCHW格式。但在这种情况下，可能会由于TensorFlow模型太复杂，导致出错。  指定`--without_data_format_optimization`后，会停止对`axis`，`shape`等参数的优化（这可能会带来一定数量的transpose操作）

**Q3. ONNX模型转换过程中，提示『Unknown shape for input tensor[tensor name: "input"] -> shape: ['batch', 'sequence']， Please define shape of input here』**  
A：该提示信息表示从ONNX的模型中获取到输入tensor(tensor名为"input:)的shape是语义象征性的['batch', 'sequence']，而不是dim为int类型的shape，从而可能会因为部分node的shape无法推理，导致转换失败。所以用户可以尝试手动在提示后输入详细的shape信息，如:-1,3,224,224  其中-1表示Batch

**Q4. Paddle模型转至ONNX模型过程中，提示『The parameter normalized of multiclass_nms OP of Paddle is False, which has diff with ONNX』**  
A: 此提示为警告信息，模型仍然会正常进行转换。Paddle中`fluid.layers.multiclass_nms`算子中提供了`normalized`参数，用于表示输入box是否进行了归一化。而ONNX中的NMS算子只支持`normalized`参数为True的情况，当你转换的模型（一般是YOLOv3模型）中该参数为`False`的情况下，转换后的模型可能会与原模型存在diff。

**Q5. Paddle模型转至ONNX模型过程中，提示『Converting this model to ONNX need with static input shape, please fix input shape of this model』**  
A: 此提示为错误信息，表示该模型的转换需要固定的输入大小:
> 1. 模型来源于PaddleX导出，可以在导出的命令中，指定--fixed_input_shape=[Height,Width]，详情可见：[PaddleX模型导出文档](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/deploy/export_model.md)。
> 2. 模型来源于PaddleDetection导出，可以在导出模型的时候，指定 TestReader.inputs_def.image_shape=[Channel,Height,Width], 详情可见：[PaddleDetection模型导出文档](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/advanced_tutorials/deploy/EXPORT_MODEL.md#设置导出模型的输入大小)。
> 3. 模型来源于自己构建，可在网络构建的`fluid.data(shape=[])`中，指定shape参数来固定模型的输入大小。
