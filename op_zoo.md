# X2Paddle支持OP列表
> 目前X2Paddle支持40+的TensorFlow OP，40+的Caffe Layer，覆盖了大部分CV分类模型常用的操作。我们在如下列表中给出了目前X2Paddle支持的全部OP。

**注：** 目前，部分OP暂未支持，如您在转换过程中出现OP不支持的情况，可自行添加或反馈给我们。欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/X2Paddle/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会及时跟进：）

## TensorFlow

| 序号 | OP | 序号 | OP |序号 | OP |序号 | OP |
|------|------|------|------|------|------|------|------|
| 1 | Relu | 2 | Relu6 | 3 | Shape | 4 | Abs |
|5 | Sigmoid | 6 | Exp | 7 | Rsqrt | 8 |swish_f32 |
|9 | Tanh | 10 | LeakyRelu | 11 | Add | 12 | RealDiv |
| 13 | Sub | 14 | Maximum | 15 | Mul | 16 | FloorDiv |
|17 | Placeholder | 18 | Const | 19 | Transpose | 20 |FusedBatchNorm|
|21 | Conv2D | 22 | BiasAdd | 23 | MaxPool | 24 | DepthwiseConv2dNative |
| 25 | Reshape | 26 | AvgPool | 27 | SplitV | 28 |  SquaredDifference|
|29 | Tile | 30 | Pack | 31 | Pad | 32 |  ResizeBilinear|
|33 | Mean | 34 | MatMul | 35 | ArgMax | 36 | StridedSlice |
| 37 | Slice | 38 | Sum | 39 | Max | 40 |  Conv2DBackpropInput|
|41 | Cast | 42 | Split | 43| Squeeze | 44 |  ResizeNearestNeighbor|
|45 | Softmax | 46 | Range | 47 | ConcatV2 | 48 | * |

## Caffe

| 序号 | OP | 序号 | OP |序号 | OP |序号 | OP |
|------|------|------|------|------|------|------|------|
| 1 | * | 2 | *| 3 |* | 4 | * |

## ONNX

| 序号 | OP | 序号 | OP |序号 | OP |序号 | OP |
|------|------|------|------|------|------|------|------|
| 1 | * | 2 | *| 3 |* | 4 | * |
