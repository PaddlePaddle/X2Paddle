---
name: X2Paddle
about: 第三方模型转换至PaddlePaddle，请说明模型的来源，模型类型（例如图像分类、目标检测等）
---

请说明模型的来源，模型类型（例如图像分类、目标检测等），应用场景（如需要转为Paddle，通过PaddleLite部署到树莓派上，满足XXX需求）

**提示 上述有详细描述的issue会让工程师更快了解用户的需求，丰富产品的功能和易用性，相关issue也会被优先处理！！！**

如有原模型文件或github链接，如方便可一并附上，方便开发人员分析。

为了便于开发人员快速定位原因，建议将模型文件上传（或通过百度网盘分享）

## 示例

模型来源： TensorFlow
模型说明： 图像分类模型，其相应repo为 https://github.com/xxxx
模型文件： 链接: https://pan.baidu.com/s/1LRzTSJwsLOul99Tj5_43wQ 密码: d3dg
转换过程出错提示如下：
```
Exception: Error happened when mapping node ['prior_box@0'] to onnx, which op_type is 'prior_box' with inputs: {'Image': ['image'], 'Input': ['elementwise_add_7.tmp_1']} and outputs: {'Boxes': ['prior_box_0.tmp_0'], 'Variances': ['prior_box_0.tmp_1']}
```
