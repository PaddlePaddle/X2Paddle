## 常见问题

**Q1. TensorFlow模型转换过程中，提示『Unknown shape for input tensor[tensor name: "input"]， Please define shape of input here』？**  
A：该提示信息表示无法从TensorFlow的pb模型中获取到输入tensor(tensor名为"input:)的shape信息，所以需要用户手动在提示后输入详细的shape信息，如None,224,224,3 其中None表示Batch。


**Q2. TensorFlow模型转换失败怎么解决?**  
A: 如果并非是由缺少OP导致，那可能是由于TensorFlow模型转换时（NHWC->NCHW格式转换导致），在这种情况下，采用如下方式进行转换，同时固化输入大小的方式，继续尝试转换，见如下命令，转换过程中，根据提示，输入相应tensor的固化shape大小。
```
x2paddle -f tensorflow -m tf.pb -s pd-model --define_input_shape
```

**Q3. ONNX模型转换过程中，提示『Unknown shape for input tensor[tensor name: "input"] -> shape: ['batch', 'sequence']， Please define shape of input here』**  
A：该提示信息表示从ONNX的模型中获取到输入tensor(tensor名为"input:)的shape是语义象征性的['batch', 'sequence']，而不是dim为int类型的shape，从而可能会因为部分node的shape无法推理，导致转换失败。所以用户可以尝试手动在提示后输入详细的shape信息，如:-1,3,224,224  其中-1表示Batch。

**Q4. 如果我的tensorflow模型是checkpoint或者SavedModel格式，怎么办？**  
A：我们提供相关文档将export_tf_model.md


**Q4. 进行动态图转换时，提示『Fail to generate inference model! Problem happend while export inference model from python code...』**
A: 此提示为无法将动态图代码转换为静态图模型，有两种可能：
> 使用动态图代码确认转换后的代码是否正确，可使用如下代码进行确认：
```
import paddle
import numpy as np
np.random.seed(6)
# ipt为输入数据
ipt = np.random.rand(1, 3, 224, 224).astype("float32")
paddle.disable_static()
# pd_model_dygraph为保存路径（其中的”/“用”.“替换）
from pd_model_dygraph.x2paddle_code import main
out =main(ipt)
```
> 若运行代码无误，则说明代码中有op不支持动转静，我们将会再未来支持；若报错，则说明pytorch2paddle转换出错，请提issue，我们将及时回复。

**Q5. 目前支持了哪些op的转换呢？**  
A: 可详见[X2Paddle支持的op列表](op_list.md)。
