# X2Paddle支持OP列表
> 目前X2Paddle支持90+的TensorFlow OP，30+的Caffe Layer，80+的ONNX OP，120+的PyTorch Aten，10+的PyTorch Prim覆盖了大部分CV分类模型常用的操作。我们在如下列表中给出了目前X2Paddle支持的全部OP。

**注：** 目前，部分OP暂未支持，如您在转换过程中出现OP不支持的情况，可自行添加或反馈给我们。欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/X2Paddle/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会及时跟进：）

## TensorFlow

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1 | Abs | 2 | Add | 3 | AddN | 4 | AddV2 |
| 5 | All | 6 | ArgMax | 7 | AvgPool | 8 | BatchMatmul |
| 9 | BatchMatmulV2 | 10 | BatchToSpaceNd | 11 | BiasAdd | 12 | Cast |
| 13 | Ceil | 14 | Concat | 15 | ConcatV2 | 16 | Const |
| 17 | Conv2D | 18 | Conv2DBackpropInput | 19 | Conv3D | 20 | DepthToSpace |
| 21 | DepthwiseConv2DNative | 22 | DivNoNan | 23 | Equal | 24 | Erf |
| 25 | Exp | 26 | ExpandDims | 27 | Fill | 28 | Floor |
| 29 | FloorDiv | 30 | FloorMod | 31 | FusedBatchnorm | 32 | FusedBatchnormV3 |
| 33 | GatherNd | 34 | GatherV2 | 35 | Greater | 36 | GreaterEqual |
| 37 | Identity | 38 | IteratorV2 | 39 | LeakyRelu | 40 | LessEqual |
| 41 | LogicalAnd | 42 | Matmul | 43 | Max | 44 | Maximum |
| 45 | MaxPool | 46 | Mean | 47 | Merge | 48 | Minimum |
| 49 | MirrorPad | 50 | Mul | 51 | Neg | 52 | NotEqual |
| 53 | OneHot | 54 | OneShotIterator | 55 | Pack | 56 | Pad |
| 57 | PadV2 | 58 | Placeholder | 59 | PlaceholderWithDefault | 60 | Pow |
| 61 | Prod | 62 | RandomUniform | 63 | Range | 64 | RealDiv |
| 65 | Relu | 66 | Relu6 | 67 | Reshape | 68 | ResizeBilinear |
| 69 | ResizeNearestNeighbor | 70 | ReverseV2 | 71 | Rsqrt | 72 | Shape |
| 73 | Sigmoid | 74 | Sign | 75 | Size | 76 | Slice |
| 77 | Softmax | 78 | Softplus | 79 | SpaceToBatchNd | 80 | Split |
| 81 | SplitV | 82 | Square | 83 | SquaredDifference | 84 | Squeeze |
| 85 | StopGradient | 86 | StridedSlice | 87 | Sub | 88 | Sum |
| 89 | Switch | 90 | Tanh | 91 | Tile | 92 | TopKV2 |
| 93 | Transpose | 94 | Unpack | 95 | Where | 96 | IteratorGetNext |
| 97 | swish_f32 | | | | | | |


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
| 1 | Abs | 2 | ArgMax | 3 | AverageOool | 4 | BatchNormalization |
| 5 | Cast | 6 | Ceil | 7 | Clip | 8 | Constant |
| 9 | ConstantOfShape | 10 | Conv | 11 | ConvTranspose | 12 | DepthToSpace |
| 13 | Div | 14 | Elu | 15 | Equal | 16 | Erf |
| 17 | Exp | 18 | Expand | 19 | Flatten | 20 | Floor |
| 21 | Gather | 22 | Gemm | 23 | GlobalAveragePool | 24 | GlobalMaxPool |
| 25 | Greater | 26 | HardSigmoid | 27 | Identity | 28 | InstanceNormalization |
| 29 | LeakyRelu | 30 | Less | 31 | Log | 32 | LogSoftmax |
| 33 | LRN | 34 | LSTM | 35 | Matmul | 36 | MaxPool |
| 37 | MaxRoiPool | 38 | Mul | 39 | NonZero | 40 | NonMaxSuppression |
| 41 | Not | 42 | OneHot | 43 | Pad | 44 | Pow |
| 45 | PRelu | 46 | Range | 47 | Reciprocal | 48 | ReduceL1 |
| 49 | ReduceL2 | 50 | ReduceMax | 51 | ReduceMean | 52 | ReduceMin |
| 53 | ReduceProd | 54 | ReduceSum | 55 | Relu | 56 | Reshape |
| 57 | Resize | 58 | RoiAlign | 59 | ScatterND | 60 | Shape |
| 61 | Shrink | 62 | Sigmoid | 63 | Sign | 64 | Size |
| 65 | Slice | 66 | Softmax | 67 | SoftPlus | 68 | SoftSign |
| 69 | Split | 70 | Sqrt | 71 | Squeeze | 72 | Sub |
| 73 | Sum | 74 | Tanh | 75 | Tile | 76 | TopK |
| 77 | Transpose | 78 | Unsqueeze | 79 | Upsample | 80 | Where |
| 81 | Add | 82 | Concat | 83 | Max | 84 | Min |
| 85 | GreaterOrEqual | 86 | GatherND | 87 | And | 88 | cos |
| 89 | Neg | 90 | SpaceToDepth | 91 | GatherElement | 92 | Sin |

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
| 57 | aten::log | 58 | aten::lt |59|aten::masked\_fill_|60|aten::masked\_fill|
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
| 113 | aten::upsample\_nearest2d | 114 | aten::split\_with\_sizes | 115 | aten::sum | 116 | aten::instance\_norm |
| 117 | aten::bitwise\_not | 118 | aten::bitwise\_xor | 119 | aten::bitwise\_and | 120 | aten::silu |
| 121 | aten::repeat\_interleave | 122 | aten::maxpool1d | 123 | aten::frobenius\_norm | 124 | aten::format |
| 125 | aten::complex | 126 | aten::real | 127 | aten::imag | 128 | aten::fft\_rfftn |
| 129 | aten::fft\_irfftn | 130  | aten::hardsigmoid | 131 | aten::hardswish |  |  |


Prim:
| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | prim::Constant | 2  | prim::data | 3  | prim::DictConstruct | 4  | prim::GetAttr |
| 5  | prim::If | 6  | prim::ListConstruct | 7  | prim::ListUnpack     | 8  | prim::Loop     |
| 9  | prim::min | 10 | prim::NumToTensor | 11 | prim::RaiseException | 12 | prim::requires\_grad |
| 13 | prim::SetAttr | 14 | prim::shape | 15 | prim::TupleConstruct | 16 | prim::TupleUnpack |
| 17 | prim::unchecked\_cast | 18 | prim::Uninitialized | ||||
