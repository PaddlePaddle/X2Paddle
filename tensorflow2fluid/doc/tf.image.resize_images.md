
## tf.image.resize_images

### [tf.image.resize_images](https://www.tensorflow.org/api_docs/python/tf/image/resize_images)
``` python
tf.image.resize_images(
    images,
    size,
    method=ResizeMethod.BILINEAR,
    align_corners=False,
    preserve_aspect_ratio=False
)
```

### [paddle.fluid.layers.image_resize](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#paddle.fluid.layers.image_resize)
``` python
paddle.fluid.layers.image_resize(
    input, 
    out_shape=None, 
    scale=None, 
    name=None, 
    resample='BILINEAR', 
    actual_shape=None, 
    align_corners=True, 
    align_mode=1
)
```

### 功能差异
#### 参数种类
TensorFlow：支持`BILINEAR`,`NEAREST`,`BICUBIC`, `AREA`四种方式；  
PaddlePaddle：支持`BILINEAR`和`NEAREST`两种方式， `align_mode`是`BILINEAR`的可选项，当为1的时候，与TensorFlow功能一致。

### 代码示例
```python
# 输入图像数据shape为[None, 3, 300, 300]
inputs = fluid.layers.data(dtype='float32', shape=[3, 300, 300], name='inputs')

# 输出shape为[3, 400, 500]
outputs = fluid.layers.image_reisze(inputs, [400, 500])
```

