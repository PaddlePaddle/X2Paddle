# X2Paddle转换库

## TensorFlow预测模型

| 模型 | 代码 | 类型 | 误差 |
|------|----------|----------|----------|
| SqueezeNet | [code](https://github.com/tensorflow/tpu/blob/master/models/official/squeezenet/squeezenet_model.py)| 视觉 | atol@1e-05, rtol@1e-05 |
| MobileNet_V1 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) | 视觉 | atol@1e-05, rtol@1e-05 |
| MobileNet_V2 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) | 视觉 | atol@1e-05, rtol@1e-05 |
| ShuffleNet | [code](https://github.com/TropComplique/shufflenet-v2-tensorflow) | 视觉 | atol@1e-05, rtol@1e-05 |
| mNASNet | [code](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) | 视觉 | atol@1e-05, rtol@1e-05 |
| EfficientNet | [code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | 视觉 | atol@1e-05, rtol@1e-05 |
| Inception_V3 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) | 视觉 | - |
| Inception_V4 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) | 视觉 | atol@1e-05, rtol@1e-05 |
| Inception_ResNet_V2 | [code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) | 视觉 | atol@1e-05, rtol@1e-05 |
| VGG16 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) | 视觉 | atol@1e-05, rtol@1e-05 |
| ResNet_V1_101 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) | 视觉 | atol@1e-05, rtol@1e-05 |
| ResNet_V2_101 | [code](https://github.com/tensorflow/models/tree/master/research/slim/nets) | 视觉 | atol@1e-05, rtol@1e-05 |
| UNet | [code1](https://github.com/jakeret/tf_unet)/[code2](https://github.com/lyatdawn/Unet-Tensorflow) | 视觉 | atol@1e-04, rtol@1e-04 |
| MTCNN | [code](https://github.com/AITTSMD/MTCNN-Tensorflow) | 视觉 | atol@1e-05, rtol@1e-05 |
| YOLO-V3| [code](https://github.com/YunYang1994/tensorflow-yolov3) | 视觉 | atol@1e-04, rtol@1e-04 |
| FALSR | [code](https://github.com/xiaomi-automl/FALSR) | 视觉 | - |
| DCSCN | [code](https://modelzoo.co/model/dcscn-super-resolution) | 视觉 | - |
| Bert（albert） | [code](https://github.com/google-research/albert#pre-trained-models) | 自然语言处理 | atol@1e-04 |
| Bert（chinese_L-12_H-768_A-12） | [code](https://github.com/google-research/bert#pre-trained-models) | 自然语言处理 | - |
| Bert（multi_cased_L-12_H-768_A-12） | [code](https://github.com/google-research/bert#pre-trained-models) |自然语言处理 | - |

【备注】`-` 代表源模型已无法获取，或未测试精度。


## Caffe预测模型

| 模型 | 代码 | 类型 | 误差 |
|------|----------|----------|----------|
| SqueezeNet | [code](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) | 视觉 | atol@1e-05, rtol@1e-05 |
| MobileNet_V1 | [code](https://github.com/shicai/MobileNet-Caffe) | 视觉 | atol@1e-05, rtol@1e-05 |
| MobileNet_V2 | [code](https://github.com/shicai/MobileNet-Caffe) | 视觉 | atol@1e-05, rtol@1e-05 |
| ShuffleNet_v2 | [code](https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe/releases/tag/v0.1.0) | 视觉 | atol@1e-05, rtol@1e-05 |
| InceptionV3 | [code](https://github.com/soeaver/caffe-model/blob/master/cls/inception/) | 视觉 | - |
| InceptionV4 | [code](https://github.com/soeaver/caffe-model/blob/master/cls/inception/) | 视觉 | - |
| mNASNet | [code](https://github.com/LiJianfei06/MnasNet-caffe) | 视觉 | atol@1e-05, rtol@1e-05 |
| MTCNN | [code](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv1/model) | 视觉 | atol@1e-05, rtol@1e-05 |
| Mobilenet_SSD | [code](https://github.com/chuanqi305/MobileNet-SSD) | 视觉 | atol@1e-05 |
| ResNet18 | [code](https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet/blob/master/deploy.prototxt) | 视觉 | atol@1e-05, rtol@1e-05 |
| ResNet50 | [code](https://github.com/soeaver/caffe-model/blob/master/cls/resnet/deploy_resnet50.prototxt) | 视觉 | atol@1e-05, rtol@1e-05 |
| Unet | [code](https://github.com/jolibrain/deepdetect/blob/master/templates/caffe/unet/deploy.prototxt) | 视觉 | atol@1e-05, rtol@1e-05 |
| VGGNet | [code](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt) | 视觉 | atol@1e-05, rtol@1e-05 |
| FaceDetection | - | 视觉 | - |

【备注】`-` 代表源模型已无法获取，或未测试精度。


## ONNX预测模型
**注：** 部分模型来源于PyTorch，PyTorch的转换可参考[pytorch_to_onnx.md](../inference_model_convertor/pytorch2onnx.md)

| 模型 | 来源 | operator version| 类型 | 误差 | 备注 |
|-------|--------|---------|---------|---------|---------|
| ResNet18 | [torchvison.model.resnet18](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| ResNet34 | [torchvison.model.resnet34](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| ResNet50 | [torchvison.model.resnet50](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| ResNet101 | [torchvison.model.resnet101](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| VGG11 | [torchvison.model.vgg11](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| VGG11_bn | [torchvison.model.vgg11_bn](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| VGG19| [torchvison.model.vgg19](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| DenseNet121 | [torchvison.model.densenet121](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| AlexNet | [torchvison.model.alexnet](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| ShuffleNet | [onnx official](https://github.com/onnx/models/tree/master/vision/classification/shufflenet) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| Inception_V2 | [onnx official](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| MobileNet_V2 | [pytorch(personal practice)](https://github.com/tonylins/pytorch-mobilenet-v2) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| mNASNet | [pytorch(personal practice)](https://github.com/rwightman/gen-efficientnet-pytorch) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| EfficientNet | [pytorch(personal practice)](https://github.com/rwightman/gen-efficientnet-pytorch) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| SqueezeNet | [onnx official](https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz) |9| 视觉 | atol@1e-05, rtol@1e-05 |
| Ultra-Light-Fast-Generic-Face-Detector-1MB| [onnx_model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/models/onnx)|9 | 视觉 | atol@1e-04, rtol@1e-04 |
| BERT | [pytorch(huggingface)](https://github.com/huggingface/notebooks/blob/master/examples/onnx-export.ipynb)|11| 自然语言处理 | atol@1e-05, rtol@1e-05 | 转换时需指定input shape，见[文档Q3](../inference_model_convertor/FAQ.md)|
| GPT2 | [pytorch(huggingface)](https://github.com/huggingface/notebooks/blob/master/examples/onnx-export.ipynb)|11| 自然语言处理 | - | 转换时需指定input shape，见[文档Q3](../inference_model_convertor/FAQ.md)|
| CifarNet | [tensorflow](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)|9|视觉 | atol@1e-05, rtol@1e-05 |
| Fcos | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py)|11| 视觉 | atol@1e-05, rtol@1e-05 |
| Yolov3 | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py)|11| 视觉 | atol@1e-05, rtol@1e-05 |
| RetinaNet | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py)|11| 视觉 | atol@1e-05, rtol@1e-05 |
| FSAF | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/fsaf/fsaf_r50_fpn_1x_coco.py)|11| 视觉 | atol@1e-05, rtol@1e-05 |
| SSD | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd/ssd300_coco.py)|11| 视觉 | atol@1e-05, rtol@1e-05 |
| Faster R-CNN | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py)|11| 视觉 | atol@1e-05, rtol@1e-05 |

【备注】`-` 代表源模型已无法获取，或未测试精度。


## PyTorch预测模型

| 模型 | 代码 | 类型 | 误差 | 备注 |
|------|----------|------|----------|------|
| AlexNet | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)| 视觉 | atol@1e-04 |-|
| MNasNet | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py) | 视觉 | atol@1e-04 |-|
| MobileNetV2 | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py) | 视觉 | atol@1e-04 |-|
| ResNet18 | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) | 视觉 | atol@1e-04 |-|
| ShuffleNetV2 | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py) | 视觉 | atol@1e-04 |-|
| SqueezeNet | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py) | 视觉 | atol@1e-04 |-|
| VGG16 | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) | 视觉 | atol@1e-04 |-|
| InceptionV3 | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py) | 视觉 | atol@1e-04 |-|
| DeepLabv3_ResNet50 | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py) | 视觉 | atol@1e-04 |-|
| FCN_ResNet50 | [code](https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/fcn.py) | 视觉 | atol@1e-04 |-|
| CamembertForQuestionAnswering | [code](https://huggingface.co/transformers/model_doc/camembert.html) | 自然语言处理 | atol@1e-04 | 只支持trace模式|
| DPRContextEncoder | [code](https://huggingface.co/transformers/model_doc/dpr.html) | 自然语言处理 | atol@1e-04 | 只支持trace模式|
| ElectraModel | [code](https://huggingface.co/transformers/model_doc/electra.html) | 自然语言处理 | atol@1e-04 | 只支持trace模式|
| FlaubertModel | [code](https://huggingface.co/transformers/model_doc/flaubert.html) | 自然语言处理 | atol@1e-04 | 只支持trace模式|
| Roberta| [code](https://huggingface.co/transformers/model_doc/roberta.html)  | 自然语言处理 | atol@1e-04 | 只支持trace模式|
| XLMRobertaForTokenClassification|[code](https://huggingface.co/transformers/model_doc/xlmroberta.html)  | 自然语言处理 | - | 只支持trace模式|
| EasyOCR_detector|[code](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/detection.py)  | 视觉 | atol@1e-04 |-|
| EasyOCR_recognizer|[code](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/recognition.py)  | 视觉 | atol@1e-04 |-|
| SwinTransformer|[code](https://github.com/microsoft/Swin-Transformer/)  | 视觉 | atol@1e-04 |-|
| BASNet|[code](https://github.com/xuebinqin/BASNet)  | 视觉 | atol@1e-04 |-|
| DBFace |[code](https://github.com/dlunion/DBFace)  | 视觉 | atol@1e-04 |-|
| SOLAR |[code](https://github.com/tonyngjichun/SOLAR)  | 自然语言处理 | - | 只支持trace模式|
| YOLOX |[code](https://github.com/Megvii-BaseDetection/YOLOX)  | 视觉 | - | 只支持trace模式|
| YOLOv5 |[code](https://github.com/ultralytics/yolov5) | 视觉 | atol@3e-03 | 只支持trace模式|
| MockingBird |[code](https://github.com/babysor/MockingBird)  | 语音 | - | 只支持trace模式|
| GPT2 |[code](https://hf-mirror.com/openai-community/gpt2/tree/main)  | 自然语言处理 | atol@1e-04 | 只支持trace模式|
| MT5_small |[code](https://hf-mirror.com/google/mt5-small/tree/main)  | 自然语言处理 | atol@2e-04 | 只支持trace模式|

【备注】`-` 代表源模型已无法获取，或未测试精度。


## PyTorch训练项目
| 模型 | 转换前代码 | 转换后代码 |
|------|----------|------|
| StarGAN | [code](https://github.com/yunjey/stargan)|[code](https://github.com/SunAhong1993/stargan/tree/paddle)|
| Ultra-Light-Fast-Generic-Face-Detector | [code](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) |[code](https://github.com/SunAhong1993/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/paddle)|

**注：** 受限于不同框架的差异，部分预测模型可能会存在目前无法转换的情况，如TensorFlow中包含控制流的模型等。对于常见的预测模型或PyTorch项目，如若您发现无法转换或转换失败，存在较大diff等问题，欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/X2Paddle/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会及时跟进。
