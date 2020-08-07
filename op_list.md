# X2Paddle支持OP列表
> 目前X2Paddle支持50+的TensorFlow OP，30+的Caffe Layer，覆盖了大部分CV分类模型常用的操作。我们在如下列表中给出了目前X2Paddle支持的全部OP。

**注：** 目前，部分OP暂未支持，如您在转换过程中出现OP不支持的情况，可自行添加或反馈给我们。欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/X2Paddle/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会及时跟进：）

## TensorFlow

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Relu        | 2  | Relu6     | 3  | Shape     | 4  | Abs                   |
| 5  | Sigmoid     | 6  | Exp       | 7  | Rsqrt     | 8  | swish_f32             |
| 9  | Tanh        | 10 | LeakyRelu | 11 | Add       | 12 | RealDiv               |
| 13 | Sub         | 14 | Maximum   | 15 | Mul       | 16 | FloorDiv              |
| 17 | Placeholder | 18 | Const     | 19 | Transpose | 20 | FusedBatchNorm        |
| 21 | Conv2D      | 22 | BiasAdd   | 23 | MaxPool   | 24 | DepthwiseConv2dNative |
| 25 | Reshape     | 26 | AvgPool   | 27 | SplitV    | 28 | SquaredDifference     |
| 29 | Tile        | 30 | Pack      | 31 | Pad       | 32 | ResizeBilinear        |
| 33 | Mean        | 34 | MatMul    | 35 | ArgMax    | 36 | StridedSlice          |
| 37 | Slice       | 38 | Sum       | 39 | Max       | 40 | Conv2DBackpropInput   |
| 41 | Cast        | 42 | Split     | 43 | Squeeze   | 44 | ResizeNearestNeighbor |
| 45 | Softmax     | 46 | Range     | 47 | ConcatV2  | 48 | MirrorPad             |
| 49 | Identity    | 50 | GreaterEqual  | 51 | StopGradient | 52 | Minimum |
| 53 | RadnomUniform | 54 | Fill | 55 | Floor | 56 | DepthToSpace |

## Caffe

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Input     | 2  | Convolution  | 3  | Deconvolution  | 4  | Pooling              |
| 5  | LRN       | 6  | InnerProduct | 7  | Softmax        | 8  | Slice                |
| 9  | Concat    | 10 | PReLU        | 11 | Accuracy       | 12 | Eltwise              |
| 13 | BatchNorm | 14 | Scale        | 15 | Reshape        | 16 | ArgMax               |
| 17 | Crop      | 18 | Flatten      | 19 | Power          | 20 | Reduction            |
| 21 | Axpy      | 22 | ROIPolling   | 23 | Permute        | 24 | DetectionOutput      |
| 25 | Normalize | 26 | Select       | 27 | ShuffleChannel | 28 | ConvolutionDepthwise |
| 29 | ReLU      | 30 | AbsVal       | 31 | Sigmoid        | 32 | TanH                 |
| 33 | ReLU6     | 34 | Upsample |

## ONNX

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Relu     | 2  | LeakyRelu | 3  | Elu       | 4  | ThresholdedRelu    |
| 5  | Prelu    | 6  | Tanh      | 7  | Shrink    | 8  | Sigmoid            |
| 9  | Pow      | 10 | Softplus  | 11 | Softsign  | 12 | HardSigmoid        |
| 13 | Exp      | 14 | Add       | 15 | Div       | 16 | Sub                |
| 17 | Mul      | 18 | Shape     | 19 | Clip      | 20 | AveragePool        |
| 21 | Sqrt     | 22 | ReduceSum | 23 | ReduceMin | 24 | ReduceMean         |
| 25 | Constant | 26 | Pad       | 27 | Unsqueeze | 28 | Resize             |
| 29 | Upsample | 30 | Expand    | 31 | Gather    | 32 | Slice              |
| 33 | Cast     | 34 | Split     | 35 | Reshape   | 36 | ConstantOfShape    |
| 37 | Ceil     | 38 | Concat    | 39 | Flatten   | 40 | ConvTranspose      |
| 41 | MatMul   | 42 | Sum       | 43 | Transpose | 44 | BatchNormalization |
| 45 | Squeeze  | 46 | Equal     | 47 | Identity  | 48 | GlobalAveragePool  |
| 49 | MaxPool  | 50 | Conv      | 51 | Gemm      | 52 | NonZero            |
| 53 | Abs      | 54 | Floor     |
