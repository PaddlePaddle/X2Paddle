# X2Paddle支持OP列表
> 目前X2Paddle支持80+的TensorFlow OP，30+的Caffe Layer，60+的ONNX OP，110+的PyTorch Aten，10+的PyTorch Prim覆盖了大部分CV分类模型常用的操作。我们在如下列表中给出了目前X2Paddle支持的全部OP。

**注：** 目前，部分OP暂未支持，如您在转换过程中出现OP不支持的情况，可自行添加或反馈给我们。欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/X2Paddle/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会及时跟进：）

## TensorFlow

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Relu             | 2  | Relu6          | 3  | Shape          | 4  | Abs                   |
| 5  | Sigmoid          | 6  | Exp            | 7  | Rsqrt          | 8  | swish_f32             |
| 9  | Tanh             | 10 | LeakyRelu      | 11 | Add            | 12 | RealDiv               |
| 13 | Sub              | 14 | Maximum        | 15 | Mul            | 16 | FloorDiv              |
| 17 | Placeholder      | 18 | Const          | 19 | Transpose      | 20 | FusedBatchNorm        |
| 21 | Conv2D           | 22 | BiasAdd        | 23 | MaxPool        | 24 | DepthwiseConv2dNative |
| 25 | Reshape          | 26 | AvgPool        | 27 | SplitV         | 28 | SquaredDifference     |
| 29 | Tile             | 30 | Pack           | 31 | Pad            | 32 | ResizeBilinear        |
| 33 | Mean             | 34 | MatMul         | 35 | ArgMax         | 36 | StridedSlice          |
| 37 | Slice            | 38 | Sum            | 39 | Max            | 40 | Conv2DBackpropInput   |
| 41 | Cast             | 42 | Split          | 43 | Squeeze        | 44 | ResizeNearestNeighbor |
| 45 | Softmax          | 46 | Range          | 47 | ConcatV2       | 48 | MirrorPad             |
| 49 | Identity         | 50 | GreaterEqual   | 51 | StopGradient   | 52 | Minimum               |
| 53 | RandomUniform    | 54 | Fill           | 55 | Floor          | 56 | DepthToSpace          |
| 57 | Sqrt             | 58 | Softplus       | 59 | Erf            | 60 | AddV2                 |
| 61 | LessEqual        | 62 | BatchMatMul    | 63 | BatchMatMulV2  | 64 | ExpandDims            |
| 65 | BatchToSpaceND   | 66 | SpaceToBatchND | 67 | OneHot         | 68 | Pow                   |
| 69 | All              | 70 | GatherV2       | 71 | IteratorV2     | 72 | Neg |
| 73 | Greater | 74 | FloorMod | 75 | LogicalAdd | 76 | Prod |
| 77 | Equal | 78 | Conv3D | 79 | Ceil | 80 | AddN |
| 81 | DivNoNan | 82 | Where | 83 | MirrorPad | 84 | Size |
| 85 | TopKv2 | 86 | SplitV |  |  |  |  |

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
| 33 | ReLU6     | 34 | Upsample     | 35 | MemoryData     |    |                      |

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
| 53 | Abs      | 54 | Floor     | 56 | ArgMax    | 57 | Sign               |
| 58 | Reciprocal  | 59 | Size     | 60 | OneHot    | 61 | ReduceProd       |
| 62 | LogSoftmax  | 63 | LSTM     | 64 |   LRN  |  |        |



## PyTorch
Aten:
| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | aten::abs | 2  | aten::adaptive_avg_pool2d | 3  | aten::addmm | 4  | aten::add |
| 5  | aten::add\_ | 6  | aten::\_\_and\_\_ | 7  | aten::append | 8  | aten::arange |
| 9  | aten::avg\_pool2d | 10 | aten::avg\_pool3d | 11 | aten::avg_pool1d | 12 | aten::batch_norm |
| 13 | aten::cat | 14 | aten::chunk | 15 | aten::clamp | 16 | aten::\_\_contains\_\_ |
| 17 | aten::constant\_pad\_nd | 18 | aten::contiguous | 19 | aten::conv2d | 20 | aten::\_convolution |
| 21 | aten::conv_transpose2d | 22 | aten::cos | 23 | aten::cumsum | 24 | aten::detach |
| 25 | aten::dict | 26 | aten::dim | 27 | aten::div\_ | 28 | aten::div   |
| 29 | aten::dropout | 30 | aten::dropout_ | 31 | aten::embedding | 32 | aten::eq     |
| 33 | aten::exp | 34 | aten::expand | 35 | aten::expand_as | 36 | aten::eye |
| 37 | aten::feature_dropout | 38 | aten::flatten | 39 | aten::Float | 40 | aten::floor |
| 41 | aten::floordiv | 42 | aten::floor_divide | 43 | aten::full_like | 44 | aten::gather |
| 45 | aten::gelu | 46 | aten::\_\_getitem\_\_ | 47 | aten::gt | 48 | aten::hardtanh\_ |
| 49 | aten::index\_select | 50 | aten::Int | 51 | aten::\_\_is\_\_ | 52 | aten::\_\_isnot\_\_ |
| 53 | aten::layer\_norm | 54 | aten::le |55|aten::leaky\_relu\_|56|aten::len|
| 57 | aten::log | 58 | aten::lt |59|aten::masked\_fil\l_|60|aten::masked\_fill|
| 61 | aten::max | 62 | aten::max\_pool2d |63|aten::matmul|64|aten\_min|
| 65 | aten::mean | 66 | aten::meshgrid |67|aten::mul|68|aten::mul\_|
| 69 | aten::ne | 70 | aten::neg |71|aten::\_\_not\_\_|72|aten::ones|
| 73 | aten::permute | 74 | aten::pow |75|aten::relu|76|aten::relu\_|
| 77 | aten::relu6 | 78 | aten::repeat |79|aten::reshape|80|aten::rsub|
| 81 | aten::ScalarImplicit | 82 | aten::select |83|aten::\_set\_item|84|aten::sigmoid|
| 85 | aten::sin | 86 | aten::size |87|aten::slice|88|aten::softmax|
| 89 | aten::softplus | 90 | aten::sqrt |91|aten::squeeze|92|aten::stack|
| 93 | aten::sub | 94 | aten::t |95|aten::tanh|96|aten::split|
| 97 | aten::transpose | 98 | aten::to |99|aten::type\_as|100|aten::unsqueeze|
| 101 | aten::upsample\_bilinear2d | 102 | aten::values |103|aten::view|104|aten::warn|
| 105 | aten::where | 106 | aten::zeros |107|aten::zeros\_like|108|aten::bmm|
| 109 | aten::sub\_ | 110 | aten:erf |111|aten::lstm|112|aten::gather|
| 113 | aten::upsample\_nearest2d | 114 |aten::split\_with\_sizes  |||||

Prim:
| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | prim::Constant | 2  | prim::data | 3  | prim::DictConstruct | 4  | prim::GetAttr |
| 5  | prim::If | 6  | prim::ListConstruct | 7  | prim::ListUnpack     | 8  | prim::Loop     |
| 9  | prim::min | 10 | prim::NumToTensor | 11 | prim::RaiseException | 12 | prim::requires\_grad |
| 13 | prim::SetAttr | 14 | prim::shape | 15 | prim::TupleConstruct | 16 | prim::TupleUnpack |
| 17 | prim::unchecked\_cast | 18 | prim::Uninitialized | ||||
