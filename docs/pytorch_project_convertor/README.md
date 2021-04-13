# PyTorch项目转换

支持将PyTorch代码及预训练模型转换为PaddlePaddle代码及预训练模型。

## 环境依赖
python >= 3.5  
paddlepaddle >= 2.0.1 或者 develop   
torch (版本由需转换的代码所需的运行环境决定)

## 安装
```
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
git checkout project_convert
python setup.py install
```

## 使用方法
### 第一步：转换前代码预处理
由于部分PyTorch操作是目前PaddlePaddle暂不支持的操作（例如：不支持TensorBoard、自动下载模型等），因此我们需要手动将这部分操作去除或者修改，具体可参见[转换前代码预处理](./before_convert.md)。



### 第二步：转换
``` shell
x2paddle --convert_torch_project --project_dir=torch_project --save_dir=paddle_project --pretrain_model=model.pth
```
| 参数 | |
|----------|--------------|
|--convert_torch_project | 当前方式为对PyTorch Project进行转换 |
|--project_dir | PyTorch的项目路径 |
|--save_dir | 指定转换后项目的保存路径 |
|--pretrain_model | **[可选]**需要转换的预训练模型的路径(文件后缀名为“.pth”、“.pt”、“.ckpt”)或者包含预训练模型的文件夹路径，转换后的模型将将保在当前路径，后缀名为“.pdiparams” |


### 第三步：转换后代码后处理
PaddlePaddle在使用上有部分限制（例如：自定义Dataset必须继承自`paddle.io.Dataset`、部分情况下DataLoader的num_worker只能为0等），用户需要手动修改代码，使代码运行，具体可参见[转换后代码后处理](./after_convert.md)。

***[注意]*** 转换前后相应操作可以参考[转换示例](./demo.md)

## Q&A
1.出现如下提示如何处理？  
> The unsupported packages or operators are: [torchvision.transforms.RandomErasing, torchvision.transforms.functional, torchvision.transforms.RandomCrop.get_params, torch.all, torch.as_tensor].  

A：这一提示说明仍有API未支持转换，用户可自行添加相应API的支持，具体添加流程参照[添加示例](./add_api.md)，或及时提issue与我们联系。 

2.运行时，出现如下2种错误，如何处理？  
> AttributeError: 'Tensor' object has no attribute 'XX'  
> AttributeError: 'Layer' object has no attribute 'XX'  

A：这一提示说明`paddle.nn.Tensor`或`paddle.nn.Layer`仍有attribute未支持转换，用户可自行添加相应API的支持，具体添加流程参照[添加示例](./add_api.md)，或及时提issue与我们联系。 


3.运行时，出现DataLoader的报错异常，如何查找原因？  
A：  
步骤一：查看对应自定义Dataset中\_\_getiem\_\_的返回值是否为numpy；  
步骤二：如若当前的设备为GPU，是否未将`num_workers`设置为0；  
步骤三：查看图像预处理的transform中是否有使用出错。  

4.当前是否支持torch.jit的转换？  
A：不支持。  

5.如何查看PyTorch与PaddlePaddle API的差异？
A：我们提供了[PyTorch-PaddlePaddle API对应表](./API_docs/README.md)，您可从中获取对应关系。