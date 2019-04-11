# caffe2fluid
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

This tool is used to convert a Caffe model to a Fluid model. In the [[doc](doc/ReadMe.md)] directory, the common APIs of Caffe-PaddlePaddle are compared and analyzed.

## Prerequisites

> python >= 2.7  
> numpy  
> protobuf >= 3.6.0  
> future 

**The running process of caffe2fluid only relies on above conditions.**  
It is recommended to install the Caffe and PaddlePaddle in the environment for testing after converting the model. For environmental installation, please refer to [Installation Documentation](prepare_en.md)

## HowTo

### Model Conversion
1. Convert the Caffe's model to the PaddlePaddle's model code and parameter file (The parameters are saved as the form of numpy).

```
# --def_path : The path of Caffe's configuration file 
# --caffemodel : The save path of Caffe's model file
# --data-output-path : The save path of the model after converting
# --code-output-path : The save path of the model code after converting
python convert.py --def_path alexnet.prototxt \
		--caffemodel alexnet.caffemodel \
		--data-output-path alexnet.npy \
		--code-output-path alexnet.py
```

2. The model network structure and parameters can be serialized as the model format supported by the PaddlePaddle framework.
```
# --model-param-path ： The save path of PaddlePaddle's serialized model
python alexnet.py --npy_path alexnet.npy --model-param-path ./fluid_model
```
Or you can specify the output of the saved model when saving.
```
# The output of model is the fc8 layer and prob layer.
python alexnet.py --npy_path alexnet.npy --model-param-path ./fluid --need-layers-name fc8,prob
```
Model loading and prediction can refer to the [official PaddlePaddle document](http://www.paddlepaddle.org/documentation/docs/en/1.3/api_guides/low_level/inference_en.html).

### Comparison of differences before and after model conversion
After the model is converted, the difference between the converted model and the original model can be compared layer by layer (**the running environment depends on caffe and paddlepaddle**)
```
# alexnet : The value of "name" in the Caffe's configuration file (.prototxt)
# ../../alexnet.prototxt : The path of Caffe's configuration file 
# ../../alexnet.caffemodel : The save path of Caffe's model file
# ../../alexnet.py : The save path of the model after converting
# ../../alexnet.npy : The save path of the model code after converting
# ./data/65.jpeg : The path of image which is need to reference
cd examples/imagenet
bash tools/diff.sh alexnet ../../alexnet.prototxt \
			../../alexnet.caffemodel \
			../../alexnet.py \
			../../alexnet.npy \
			./data/65.jpeg
```



## How to convert custom layer
In the model conversion, when encounter an unsupported custom layer, users can add code to achieve a custom layer according to their needs. thus supporting the complete conversion of the model. The implementation is the following process.    

1. Implement your custom layer in a file under `kaffe/custom_layers`, eg: mylayer.py
    - Implement ```shape_func(input_shape, [other_caffe_params])``` to calculate the output shape
    - Implement ```layer_func(inputs, name, [other_caffe_params])``` to construct a fluid layer
    - Register these two functions ```register(kind='MyType', shape=shape_func, layer=layer_func)```
    - Notes: more examples can be found in `kaffe/custom_layers`

2. Add ```import mylayer``` to  `kaffe/custom_layers/__init__.py`

3. Prepare your pycaffe as your customized version(same as previous env prepare)
    - (option1) replace `proto/caffe.proto` with your own caffe.proto and compile it
    - (option2) change your `pycaffe` to the customized version

4. Convert the Caffe model to Fluid model

5. Set env $CAFFE2FLUID_CUSTOM_LAYERS to the parent directory of 'custom_layers'
   ```
   export CAFFE2FLUID_CUSTOM_LAYERS=/path/to/caffe2fluid/kaffe
   ```

### Tested models
The caffe2fluid passed the test on the following model:
- Lenet:
[model addr](https://github.com/ethereon/caffe-tensorflow/blob/master/examples/mnist)

- ResNets:(ResNet-50, ResNet-101, ResNet-152)
[model addr](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)

- GoogleNet:
[model addr](https://gist.github.com/jimmie33/7ea9f8ac0da259866b854460f4526034)

- VGG:
[model addr](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)

- AlexNet:
[model addr](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)

### Notes
Some of this code come from here: [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)
