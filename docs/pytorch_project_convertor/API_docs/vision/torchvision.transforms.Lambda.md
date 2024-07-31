## torchvision.transforms.Lambda
### [torchvision.transforms.Lambda](https://pytorch.org/vision/stable/transforms.html?highlight=lambda#torchvision.transforms.Lambda)
```python
torchvision.transforms.Lambda(lambd)
```

### 功能介绍
用于使用lamda定义的函数对数据进行预处理，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。
```python
import paddle
from paddle.vision.transforms import BaseTransform
class Lambda(BaseTransform):
    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def _apply_image(self, img):
        return self.lambd(img)
```
