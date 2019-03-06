
## tf.image.non_max_suppression

### [tf.image.non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
``` python
tf.image.non_max_suppression(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
    name=None
)
```

### [paddle.fluid.layers.multiclass_nms](http://paddlepaddle.org/documentation/docs/en/1.3/api/layers.html#permalink-245-multiclass_nms)
``` python
paddle.fluid.layers.multiclass_nms(
    bboxes, 
    scores, 
    score_threshold, 
    nms_top_k, 
    keep_top_k, 
    nms_threshold=0.3, 
    normalized=True, 
    nms_eta=1.0, 
    background_label=0, 
    name=None)
```

### 功能差异
#### 输入格式
TensorFlow：`boxes`的shape为`[num_boxes, 4]`， `scores`的shape为`[num_boxes]`
PaddlePaddle：相对比Tensorflow，还支持batch和多类别，`bboxes`的shape为`[batch, num_boxes, 4]`， `scores`的shape为`[batch, num_classes, num_boxes]`

#### 输出格式
TensorFlow: 返回shape为`[N]`的tensor，表示为`boxes`中选取的index集合，长度为`N`
PaddlePaddle: 返回`[N, 6]`的[LodTensor](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/fluid_cn.html#lodtensor)，其中每行内容为`[lable, confidence, xmin, ymin, xmax, ymax]`

#### 参数差异
TensorFlow: 在所有boxes中，根据其它参数条件，最终选出的boxes数量不超过`max_output_size`  
PaddlePaddle: 在`nms_top_k`个boxes中，根据其它参数条件，最终选出的boxes数量不超过`keep_top_k`

## paddlepaddle示例:
```python
clip_boxes = fluid.layers.data(dtype='float32', shape=[5000, 4], name='boxes')
scores = fluid.layers.data(dtype='float32', shape=[1, 5000], name='scores')
selected_boxes = fluid.layers.multiclass_nms(clip_boxes, scores, scrore_threshold=0.5, nms_top
```
