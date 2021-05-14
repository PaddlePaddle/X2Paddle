## 常见问题
1.出现如下提示如何处理？  
> The no support Api are: [torchvision.transforms.RandomErasing, torchvision.transforms.functional, torchvision.transforms.RandomCrop.get_params, torch.all, torch.as_tensor].  

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
