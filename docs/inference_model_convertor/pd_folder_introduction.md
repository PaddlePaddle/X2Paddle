# 转换后PaddlePaddle预测模型简介
- 静态图：
转换后的模型包括`model_with_code`和`inference_model`两个目录。
`model_with_code`中保存了模型参数，和转换后的python模型静态图代码。
`inference_model`中保存了序列化的模型结构和参数，可直接使用paddle的接口进行加载，见[paddle.static.load_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/static/load_inference_model_cn.html#load-inference-model)。
- 动态图：
转换后的模型包括`model.pdparams`和`x2paddle_code.py`两个文件，以及`inference_model`一个目录。
`model.pdparams`中保存了模型参数。
`x2paddle_code.py`是转换后的python模型动态图代码。
`inference_model`中保存了序列化的模型结构和参数，可直接使用paddle的接口进行加载，见[paddle.static.load_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/static/load_inference_model_cn.html#load-inference-model)。
