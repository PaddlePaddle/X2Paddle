### PaddleLite部署
使用X2Paddle转换后的模型均可以使用Paddle Fluid进行预测。但对于PaddleLite上的部署，则需要检查模型中的OP是否都在PaddleLite中支持。使用`check_for_lite.py`可以进行检查。

```
python tools/check_for_lite.py paddle_model/inference_model/__model__
```
