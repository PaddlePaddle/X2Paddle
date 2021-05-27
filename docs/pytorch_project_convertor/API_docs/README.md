# PyTorch-PaddlePaddle API映射表
本文档基于[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)研发过程梳理了PyTorch（v1.8.1)常用API与PaddlePaddle 2.0.0 API对应关系与差异分析。通过本文档，帮助开发者快速迁移PyTorch使用经验，完成模型的开发与调优。

## X2Paddle介绍
X2Paddle致力于帮助其它主流深度学习框架开发者快速迁移至飞桨框架，目前提供三大功能
- 预测模型转换
  - 支持Caffe/TensorFlow/ONNX/PyTorch的模型一键转为飞桨的预测模型，并使用PaddleInference/PaddleLite进行CPU/GPU/Arm等设备的部署
- PyTorch训练项目转换
  - 支持PyTorch项目Python代码（包括训练、预测）一键转为基于飞桨框架的项目代码，帮助开发者快速迁移项目，并可享受AIStudio平台对于飞桨框架提供的海量免费计算资源
- API映射文档
  - 详细的API文档对比分析，帮助开发者快速从PyTorch框架的使用迁移至飞桨框架的使用，大大降低学习成本
 
详细的项目信息与使用方法参考X2Paddle在Github上的开源项目: https://github.com/PaddlePaddle/X2Paddle

## API映射表目录

| 类别         | 简介 |
| ---------- | ------------------------- |
| [基础操作类](https://github.com/PaddlePaddle/X2Paddle/tree/develop/docs/pytorch_project_convertor/API_docs/ops/README.md) | 主要为`torch.XX`类API |
| [组网类](https://github.com/PaddlePaddle/X2Paddle/tree/develop/docs/pytorch_project_convertor/API_docs/nn/README.md)    | 主要为`torch.nn.XX`类下组网相关的API |
| [Loss类](https://github.com/PaddlePaddle/X2Paddle/tree/develop/docs/pytorch_project_convertor/API_docs/loss/README.md) |主要为`torch.nn.XX`类下loss相关的API    |
|  [工具类](https://github.com/PaddlePaddle/X2Paddle/tree/develop/docs/pytorch_project_convertor/API_docs/utils/README.md)   | 主要为`torch.nn.XX`类下分布式相关的API和`torch.utils.XX`类API|
|   [视觉类](https://github.com/PaddlePaddle/X2Paddle/tree/develop/docs/pytorch_project_convertor/API_docs/vision/README.md)  | 主要为`torchvision.XX`类API |
