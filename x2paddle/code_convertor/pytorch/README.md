# PyTorch项目转换

支持将PyTorch代码及预训练模型转换为PaddlePaddle代码及预训练模型。

## 环境依赖
python >= 3.5
paddlepaddle >= 2.0.1 或者 develop
torch (版本由需转换的代码所需的运行环境决定)

## 使用方法
### 转换前操作
1. 去除TensorBoard相关的操作。

2. 将PyTorch中Tensor逐位逻辑与、或、异或运算操作符替换为对应的API的操作：
> | 替换为 torch.bitwise_or
> & 替换为 torch.bitwise_and
> ^ 替换为 toech.bitwise_xor

3. 若自定义的`DataSet`（用于加载数据模块，作为`torch.utils.data.DataLoader`的参数）未继承`torch.utils.data.Dataset`，则需要添加该继承关系。

4. 若预训练模型需要下载，去除下载预训练模型相关代码，在转换前将预训练模型下载至本地，并修改加载预训练模型参数相关代码的路径为预训练模型本地保存路径。

### 转换
```
x2paddle --framework=pytorch_project --project_dir=torch_project --save_dir=paddle_project --pretrain_model=model.pth
```
| 参数 | |
|----------|--------------|
|--framework | 当使用该方式进行转换时有且只能使用“pytorch_project” |
|--project_dir | PyTorch的项目路径 |
|--save_dir | 指定转换后项目的保存路径 |
|--pretrain_model | **[可选]**需要转换的预训练模型的路径(文件后缀名为“.pth”、“.pt”、“.ckpt”)，转换后的模型将将保在当前路径，后缀名为“.pdiparams” |

### 转换后操作
1. 若需要使用GPU，`x2paddle.torch2paddle.DataLoader`中的`num_workers`必须设置为0。

2. 修改自定义Datase（继承自`paddle.io.DataSet`）t中的`__getitem__`的返回值，若返回值中存在Tensor，需添加相应代码将Tensor修改为numpy。

3. 若存在Tensor对比操作（包含==、!=、<、<=、>、>=操作符）,在对比操作符前添加对Tensor类型的判断，如果为bool型强转为int型，并在对比后转换回bool型。

***[注意]*** 转换前后相应操作可以参考[转换示例](./x2paddle/code_convertor/pytorch/PyTorchProject2Paddle_Demo.md)

## Q&A
1. 出现如下提示怎么办？
> The no support Api are: [torchvision.transforms.RandomErasing, torchvision.transforms.functional, torchvision.transforms.RandomCrop.get_params, torch.all, torch.as_tensor].
A：可自行添加相应API的支持，具体添加流程参照[添加示例](./x2paddle/code_convertor/pytorch/Add_API.md)
