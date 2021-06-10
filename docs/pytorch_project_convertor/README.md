# PyTorch训练项目转换

支持将PyTorch代码及预训练模型转换为PaddlePaddle代码及预训练模型。

## 使用方法
### 第一步：转换前代码预处理
由于部分PyTorch操作是目前PaddlePaddle暂不支持的操作（例如：不支持TensorBoard、自动下载模型等），因此我们需要手动将这部分操作去除或者修改，具体可参见[转换前代码预处理](./before_convert.md)。



### 第二步：转换
``` shell
x2paddle --convert_torch_project --project_dir=torch_project --save_dir=paddle_project --pretrain_model=model.pth
```
| 参数 | 作用|
|----------|--------------|
|--convert_torch_project | 当前方式为对PyTorch Project进行转换 |
|--project_dir | PyTorch的项目路径 |
|--save_dir | 指定转换后项目的保存路径 |
|--pretrain_model | **[可选]**需要转换的预训练模型的路径(文件后缀名为“.pth”、“.pt”、“.ckpt”)或者包含预训练模型的文件夹路径，转换后的模型将将保在当前路径，后缀名为“.pdiparams” |


### 第三步：转换后代码后处理
PaddlePaddle在使用上有部分限制（例如：自定义Dataset必须继承自`paddle.io.Dataset`、部分情况下DataLoader的num_worker只能为0等），用户需要手动修改代码，使代码运行，具体可参见[转换后代码后处理](./after_convert.md)。

***[注意]*** 转换前后相应操作可以参考[转换示例](./demo/README.md)

## 致谢
感谢[aiyasin](https://github.com/aiyasin)为本文档贡献issue和PR，同时也感谢[jstzwjr](https://github.com/jstzwjr)、[faded-TJU](https://github.com/faded-TJU)、[freemustard](https://github.com/freemustard)、[156aasdfg](https://github.com/156aasdfg)、
[hrdwsong](https://github.com/hrdwsong)、[geoyee](https://github.com/geoyee)、[ArlanCooper](https://github.com/ArlanCooper)、
[Felix-python](https://github.com/Felix-python)、[2U-maker](https://github.com/2U-maker) 、[AlexZou14](https://github.com/AlexZou14)、[skywalk163](https://github.com/skywalk163)、[Darki-luo](https://github.com/Darki-luo)在论文复现营比赛过程中通过issue积极为X2Paddle反馈使用中存在的问题！
