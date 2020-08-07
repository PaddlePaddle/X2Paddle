目前，代码中已经提供了10个非官方op（不在[官网](http://caffe.berkeleyvision.org/tutorial/layers)上的op）的转换，这些op对应的Caffe实现源码如下：

| op | 该版本实现源码 |
|-------|--------|
| PriorBox | [code](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp) |
| DetectionOutput | [code](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/detection_output_layer.cpp) |
| ConvolutionDepthwise | [code](https://github.com/farmingyard/caffe-mobilenet/blob/master/conv_dw_layer.cpp) |
| ShuffleChannel | [code](https://github.com/farmingyard/ShuffleNet/blob/master/shuffle_channel_layer.cpp) |
| Permute | [code](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/permute_layer.cpp) |
| Normalize | [code](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/normalize_layer.cpp) |
| ROIPooling | [code](https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/roi_pooling_layer.cpp) |
| Axpy | [code](https://github.com/hujie-frank/SENet/blob/master/src/caffe/layers/axpy_layer.cpp) |
| ReLU6 | [code](https://github.com/chuanqi305/ssd/blob/ssd/src/caffe/layers/relu6_layer.cpp) |
| Upsample | [code](https://github.com/eric612/MobileNet-YOLO/blob/master/src/caffe/layers/upsample_layer.cpp) |
