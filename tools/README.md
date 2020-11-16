### 一、PaddleLite部署
使用X2Paddle转换后的模型均可以使用Paddle Fluid进行预测。但对于PaddleLite上的部署，则需要检查模型中的OP是否都在PaddleLite中支持。使用`check_for_lite.py`可以进行检查。

```
python tools/check_for_lite.py paddle_model/inference_model/__model__
```
> 附：check_for_lite工具并不能完全判断模型是否被支持，PaddleLite详细支持的算子请参考[PaddleLite支持算子集](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/introduction/support_operation_list.md)


### 二、模型参数合并
X2Paddle转换后产出的路径下包括两个目录，  
1. `model_with_code`: 包含保存的参数文件和模型python代码文件，供用户debug  
2. `inference_model`: 参数文件和序列化的模型结构文件，供用户预测部署  

其中在`inference_model`中，X2Paddle将每个参数独立保存在不同的文件中（文件名和参数名一致），用户可使用`merge_params.py`将参数文件合并成一个文件使用
```
python tools/merge_params.py paddle_model/inference_model  new_model_dir
```
合并参数后的模型保存在`new_model_dir`中
