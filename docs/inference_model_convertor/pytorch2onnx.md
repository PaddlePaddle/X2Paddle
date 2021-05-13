## PyTorch模型导出为ONNX模型

目前onnx2paddle主要支持onnx operator version 9。 用户可通过如下示例代码，将torchvison或者自己开发写的模型转换成onnx model:
```
#coding: utf-8
import torch
import torchvision

# 指定输入大小的shape
dummy_input = torch.randn(1, 3, 224, 224)

# 构建pytorch model，并载入模型参数
resnet18 = torchvision.models.resnet18(pretrained=True)

# 导出resnet18.onnx模型文件
torch.onnx.export(resnet18, dummy_input, "resnet18.onnx",verbose=True)

```
