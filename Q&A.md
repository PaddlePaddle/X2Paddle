常见问题

**Q1. TensorFlow模型转换过程中，提示『Unknown shape for input tensor[tensor name: "input"]， Please define shape of input here』？**
A：该提示信息表示无法从TensorFlow的pb模型中获取到输入tensor(tensor名为"input:)的shape信息，所以需要用户手动在提示后输入详细的shape信息，如None,224,224,3 其中None表示Batch 
