# Tensor类API关键字参数
PyTorch中生成Tensor类API主要有`torch.full`、`torch.arange`、`torch.range`、`torch.eye`、`torch.linspace`，这类API关键字参数只存在于PyTorch API中，PaddlePaddle对应API中不存在。关键字参数在下述表格中说明：
| 参数名            | 参数含义                                              |
| ----------------- | ----------------------------------------------------- |
| **out**           | 输出的Tensor。                                        |
| **layout**        | 输出Tensor所需要的布局。                              |
| **device**        | 输出Tensor所存放的设备。                              |
| **requires_grad** | 是否阻断Autograd的梯度传导，默认为False，代表不阻断。 |