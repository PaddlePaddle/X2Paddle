# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from x2paddle.project_convertor.pytorch.api_mapper import *
from x2paddle.utils import is_new_version

OPTIMIZER_MAPPER = {
    "torch.optim": ["paddle.optimizer", None],
    "torch.optim.lr_scheduler.ReduceLROnPlateau":
    ["paddle.optimizer.lr.ReduceOnPlateau", LRScheculerMapper],
    "torch.optim.lr_scheduler.CosineAnnealingLR":
    ["paddle.optimizer.lr.CosineAnnealingDecay", LRScheculerMapper],
    "torch.optim.lr_scheduler.MultiStepLR":
    ["paddle.optimizer.lr.MultiStepDecay", LRScheculerMapper],
    "torch.optim.Adam": ["x2paddle.torch2paddle.Adam", None],
    "torch.optim.SGD": ["x2paddle.torch2paddle.Momentum", None]
}

NN_MAPPER = {
    # basic
    "torch.nn": ["paddle.nn", None],
    "torch.nn.Module": ["paddle.nn.Layer", None],
    "torch.nn.ModuleList": ["paddle.nn.LayerList", None],
    "torch.nn.Sequential": ["paddle.nn.Sequential", None],
    "torch.nn.utils": ["paddle.nn.utils", None],
    "torch.nn.utils.clip_grad_value_":
    ["x2paddle.torch2paddle.clip_grad_value_", None],
    "torch.nn.utils.spectral_norm":
    ["x2paddle.torch2paddle.spectral_norm", None],
    "torch.nn.Parameter": ["paddle.create_parameter", CreateParamModuleMapper],
    "torch.nn.parallel": ["paddle", None],
    "torch.nn.DataParallel": ["paddle.DataParallel", DataParallelModuleMapper],
    "torch.nn.parallel.DistributedDataParallel":
    ["paddle.DataParallel", DataParallelModuleMapper],
    "torch.nn.functional": ["paddle.nn.functional", None],
    # nn_net
    "torch.nn.AdaptiveAvgPool1d": ["paddle.nn.AdaptiveAvgPool1D", None],
    "torch.nn.AdaptiveAvgPool2d": ["paddle.nn.AdaptiveAvgPool2D", None],
    "torch.nn.AdaptiveAvgPool3d": ["paddle.nn.AdaptiveAvgPool3D", None],
    "torch.nn.AvgPool1d": ["paddle.nn.AvgPool1D", AvgPoolModuleMapper],
    "torch.nn.AvgPool2d": ["paddle.nn.AvgPool2D", AvgPoolModuleMapper],
    "torch.nn.AvgPool3d": ["paddle.nn.AvgPool3D", AvgPoolModuleMapper],
    "torch.nn.BatchNorm1d": ["paddle.nn.BatchNorm1D", BatchNormModuleMapper],
    "torch.nn.BatchNorm2d": ["paddle.nn.BatchNorm2D", BatchNormModuleMapper],
    "torch.nn.BatchNorm3d": ["paddle.nn.BatchNorm3D", BatchNormModuleMapper],
    "torch.nn.ConstantPad2d": ["paddle.nn.Pad2D", PadModuleMapper],
    "torch.nn.Conv1d": ["paddle.nn.Conv1D", ConvModuleMapper],
    "torch.nn.Conv2d": ["paddle.nn.Conv2D", ConvModuleMapper],
    "torch.nn.Conv3d": ["paddle.nn.Conv3D", ConvModuleMapper],
    "torch.nn.ConvTranspose2d":
    ["x2paddle.torch2paddle.Conv2DTranspose", None],
    "torch.nn.Dropout": ["paddle.nn.Dropout", DropoutModuleMapper],
    "torch.nn.Dropout2d": ["paddle.nn.Dropout", DropoutModuleMapper],
    "torch.nn.Embedding": ["paddle.nn.Embedding", EmbeddingModuleMapper],
    "torch.nn.GELU": ["paddle.nn.GELU", None],
    "torch.nn.GroupNorm": ["paddle.nn.GroupNorm", None],
    "torch.nn.Identity": ["x2paddle.torch2paddle.Identity", None],
    "torch.nn.InstanceNorm2d":
    ["paddle.nn.InstanceNorm2D", BatchNormModuleMapper],
    "torch.nn.LeakyReLU": ["paddle.nn.LeakyReLU", None],
    "torch.nn.LayerNorm": ["paddle.nn.LayerNorm", BatchNormModuleMapper],
    "torch.nn.Linear": ["paddle.nn.Linear", LinearModuleMapper],
    "torch.nn.MaxPool1d": ["paddle.nn.MaxPool1D", MaxPoolModuleMapper],
    "torch.nn.MaxPool2d": ["paddle.nn.MaxPool2D", MaxPoolModuleMapper],
    "torch.nn.MaxPool3d": ["paddle.nn.MaxPool3D", MaxPoolModuleMapper],
    "torch.nn.MaxUnpool2d": ["x2paddle.torch2paddle.MaxUnpool2d", None],
    "torch.nn.ReflectionPad2d": ["paddle.nn.Pad2D", PadModuleMapper],
    "torch.nn.ReplicationPad2d": ["paddle.nn.Pad2D", PadModuleMapper],
    "torch.nn.PReLU": ["paddle.nn.PReLU", None],
    "torch.nn.ReLU": ["paddle.nn.ReLU", ReLUModuleMapper],
    "torch.nn.ReLU6": ["paddle.nn.ReLU6", ReLUModuleMapper],
    "torch.nn.Sigmoid": ["paddle.nn.Sigmoid", None],
    "torch.nn.Softmax": ["paddle.nn.Softmax", SoftmaxModuleMapper],
    "torch.nn.SyncBatchNorm":
    ["paddle.nn.SyncBatchNorm", BatchNormModuleMapper],
    "torch.nn.Tanh": ["paddle.nn.Tanh", None],
    "torch.nn.Upsample": ["paddle.nn.Upsample", None],
    "torch.nn.ZeroPad2d": ["paddle.nn.Pad2D", PadModuleMapper],
    # nn_loss
    "torch.nn.CrossEntropyLoss":
    ["paddle.nn.CrossEntropyLoss", LossModuleMapper],
    "torch.nn.BCEWithLogitsLoss":
    ["paddle.nn.BCEWithLogitsLoss", LossModuleMapper],
    "torch.nn.BCELoss": ["paddle.nn.BCELoss", None],
    "torch.nn.KLDivLoss": ["x2paddle.torch2paddle.KLDivLoss", None],
    "torch.nn.L1Loss": ["paddle.nn.loss.L1Loss", LossModuleMapper],
    # functional_net
    "torch.nn.functional.avg_pool1d":
    ["paddle.nn.functional.avg_pool1d", AvgPoolFuncMapper],
    "torch.nn.functional.avg_pool2d":
    ["paddle.nn.functional.avg_pool2d", AvgPoolFuncMapper],
    "torch.nn.functional.avg_pool3d":
    ["paddle.nn.functional.avg_pool3d", AvgPoolFuncMapper],
    "torch.nn.functional.dropout":
    ["paddle.nn.functional.dropout", DropoutFuncMapper],
    "torch.nn.functional.interpolate":
    ["paddle.nn.functional.interpolate", InterpolateFuncMapper],
    "torch.nn.functional.leaky_relu":
    ["paddle.nn.functional.leaky_relu", LeaklyReluFuncMapper],
    "torch.nn.functional.log_softmax":
    ["paddle.nn.functional.log_softmax", LogSoftmaxFuncMapper],
    "torch.nn.functional.pad": ["paddle.nn.functional.pad", PadFuncMapper],
    "torch.nn.functional.relu": ["paddle.nn.functional.relu", ReluFuncMapper],
    "torch.sigmoid": ["paddle.nn.functional.sigmoid", SigmoidFuncMapper],
    "torch.nn.functional.sigmoid":
    ["paddle.nn.functional.sigmoid", SigmoidFuncMapper],
    "torch.nn.functional.softmax":
    ["paddle.nn.functional.softmax", SoftmaxFuncMapper],
    "torch.nn.functional.tanh": ["paddle.tanh", None],
    # init
    "torch.nn.init": ["x2paddle.torch2paddle", None],
    "torch.nn.init.kaiming_normal_":
    ["x2paddle.torch2paddle.kaiming_normal_", None],
    "torch.nn.init.kaiming_normal":
    ["x2paddle.torch2paddle.kaiming_normal_", None],
    "torch.nn.init.xavier_uniform_":
    ["x2paddle.torch2paddle.xavier_normal_", None],
    "torch.nn.init.xavier_normal_":
    ["x2paddle.torch2paddle.xavier_uniform_", None],
    "torch.nn.init.constant_": ["x2paddle.torch2paddle.constant_init_", None],
    "torch.nn.init.normal_": ["x2paddle.torch2paddle.normal_init_", None],
    "torch.nn.init.ones_": ["x2paddle.torch2paddle.ones_init_", None],
    "torch.nn.init.zeros_": ["x2paddle.torch2paddle.zeros_init_", None],
    "torch.nn.init.orthogonal_":
    ["x2paddle.torch2paddle.normal_init_", None],  # syf(TODO)
    # functional_loss
    "torch.nn.functional.binary_cross_entropy_with_logits":
    ["x2paddle.torch2paddle.binary_cross_entropy_with_logits", None],
    "torch.nn.functional.cross_entropy":
    ["paddle.nn.functional.cross_entropy", CrossEntropyFuncMapper],
    "torch.nn.functional.mse_loss":
    ["paddle.nn.functional.mse_loss", LossFuncMapper],
    "torch.nn.functional.smooth_l1_loss":
    ["paddle.nn.functional.smooth_l1_loss", LossFuncMapper],
}

UTILS_MAPPER = {
    "torch.utils.data": ["paddle.io", None],
    "torch.utils.data.DataLoader": ["x2paddle.torch2paddle.DataLoader", None],
    "torch.utils.data.random_split":
    ["x2paddle.torch2paddle.random_split", None],
    "torch.utils.data.Dataset": ["paddle.io.Dataset", None],
    "torch.utils.data.ConcatDataset":
    ["x2paddle.torch2paddle.ConcatDataset", None],
    "torch.utils.data.distributed": ["x2paddle.torch2paddle", None],
    "torch.utils.data.distributed.DistributedSampler":
    ["x2paddle.torch2paddle.DistributedSampler", None],
    "torch.utils.model_zoo": ["paddle", None],
    "torch.utils.model_zoo.load_url": ["paddle.load", HubLoadMapper],
}

DIST_MAPPER = {
    "torch.multiprocessing": ["paddle.distributed", None],
    "torch.multiprocessing.spawn": ["paddle.distributed.spawn", None],
    "torch.distributed": ["x2paddle.torch2paddle", None],
    "torch.distributed.init_process_group":
    ["x2paddle.torch2paddle.init_process_group", None]
}

if is_new_version:
    DTYPE_MAPPER = {
        "torch.float16": ["paddle.float16", None],
        "torch.half": ["paddle.float16", None],
        "torch.float32": ["paddle.float32", None],
        "torch.float": ["paddle.float32", None],
        "torch.float64": ["paddle.float64", None],
        "torch.double": ["paddle.float64", None],
        "torch.uint8": ["paddle.uint8", None],
        "torch.int8": ["paddle.int8", None],
        "torch.int16": ["paddle.int16", None],
        "torch.short": ["paddle.int16", None],
        "torch.int32": ["paddle.int32", None],
        "torch.int": ["paddle.int32", None],
        "torch.int64": ["paddle.int64", None],
        "torch.long": ["paddle.int64", None],
        "torch.bool": ["paddle.bool", None],
    }
else:
    DTYPE_MAPPER = {
        "torch.float16": [string("float16"), None],
        "torch.half": [string("float16"), None],
        "torch.float32": [string("float32"), None],
        "torch.float": [string("float32"), None],
        "torch.float64": [string("float64"), None],
        "torch.double": [string("float64"), None],
        "torch.uint8": [string("uint8"), None],
        "torch.int8": [string("int8"), None],
        "torch.int16": [string("int16"), None],
        "torch.short": [string("int16"), None],
        "torch.int32": [string("int32"), None],
        "torch.int": [string("int32"), None],
        "torch.int64": [string("int64"), None],
        "torch.long": [string("int64"), None],
        "torch.bool": [string("bool"), None],
    }

TORCHVISION_MAPPER = {
    "torchvision": ["paddle.vision", None],
    # transforms
    "torchvision.transforms": ["paddle.vision.transforms", None],
    "torchvision.transforms.Compose":
    ["paddle.vision.transforms.Compose", None],
    "torchvision.transforms.ToPILImage":
    ["x2paddle.torch2paddle.ToPILImage", None],
    "torchvision.transforms.Resize": ["paddle.vision.transforms.Resize", None],
    "torchvision.transforms.ToTensor":
    ["x2paddle.torch2paddle.ToTensor", None],
    "torchvision.transforms.RandomHorizontalFlip":
    ["paddle.vision.transforms.RandomHorizontalFlip", None],
    "torchvision.transforms.CenterCrop":
    ["paddle.vision.transforms.CenterCrop", None],
    "torchvision.transforms.Normalize":
    ["x2paddle.torch2paddle.Normalize", None],
    "torchvision.transforms.RandomResizedCrop":
    ["paddle.vision.transforms.RandomResizedCrop", None],
    "torchvision.transforms.Lambda": ["x2paddle.torch2paddle.Lambda", None],
    # utils
    "torchvision.utils": ["x2paddle.torch2paddle", None],
    "torchvision.utils.save_image": ["x2paddle.torch2paddle.save_image", None],
    # datasets
    "torchvision.datasets": ["paddle.vision.datasets", None],
    "torchvision.datasets.ImageFolder":
    ["x2paddle.torch2paddle.ImageFolder", None],
    # models
    "torchvision.models": ["x2paddle.models", None],
    "torchvision.models.vgg.model_urls":
    ["x2paddle.models.vgg_pth_urls", None],
    "torchvision.models.vgg11": ["x2paddle.models.vgg11_pth", None],
    "torchvision.models.vgg13": ["x2paddle.models.vgg13_pth", None],
    "torchvision.models.vgg16": ["x2paddle.models.vgg16_pth", None],
    "torchvision.models.vgg19": ["x2paddle.models.vgg19_pth", None],
    "torchvision.models.vgg11_bn": ["x2paddle.models.vgg11_bn_pth", None],
    "torchvision.models.vgg13_bn": ["x2paddle.models.vgg13_bn_pth", None],
    "torchvision.models.vgg16_bn": ["x2paddle.models.vgg16_bn_pth", None],
    "torchvision.models.vgg19_bn": ["x2paddle.models.vgg19_bn_pth", None],
    "torchvision.models.resnet.model_urls":
    ["x2paddle.models.resnet_pth_urls", None],
    "torchvision.models.resnet18": ["x2paddle.models.resnet18_pth", None],
    "torchvision.models.resnet34": ["x2paddle.models.resnet34_pth", None],
    "torchvision.models.resnet50": ["x2paddle.models.resnet50_pth", None],
    "torchvision.models.resnet101": ["x2paddle.models.resnet101_pth", None],
    "torchvision.models.resnet152": ["x2paddle.models.resnet152_pth", None],
    "torchvision.models.resnext50_32x4d":
    ["x2paddle.models.resnext50_32x4d_pth", None],
    "torchvision.models.resnext101_32x8d":
    ["x2paddle.models.resnext101_32x8d_pth", None],
    "torchvision.models.wide_resnet50_2":
    ["x2paddle.models.wide_resnet50_2_pth", None],
    "torchvision.models.wide_resnet101_2":
    ["x2paddle.models.wide_resnet101_2_pth", None],
}

AUTOGRAD_MAPPER = {
    "torch.autograd.Variable": ["paddle.to_tensor", None],  # TODO(syf): 确认是否一致
    "torch.autograd.grad": ["paddle.grad", None],
}

API_MAPPER = {
    "torch": ["paddle", None],
    "torch.Tensor": ["x2paddle.torch2paddle.create_tensor", None],
    "torch.FloatTensor": ["x2paddle.torch2paddle.create_float32_tensor", None],
    "torch.cuda.FloatTensor":
    ["x2paddle.torch2paddle.create_float32_tensor", None],
    "torch.ByteTensor": ["x2paddle.torch2paddle.create_uint8_tensor", None],
    "torch.cuda.ByteTensor":
    ["x2paddle.torch2paddle.create_uint8_tensor", None],
    "torch.load": ["paddle.load", LoadMapper],
    "torch.save": ["paddle.save", SaveMapper],
    "torch.device": ["paddle.set_device", SetDeviceMapper],
    "torch.cat": ["x2paddle.torch2paddle.concat", None],
    "torch.cuda.is_available": ["paddle.is_compiled_with_cuda", None],
    "torch.cuda.set_device": ["x2paddle.torch2paddle.set_cuda_device", None],
    "torch.no_grad": ["paddle.no_grad", None],
    "torch.from_numpy": ["paddle.to_tensor", None],
    "torch.cuda.device_count": ["x2paddle.torch2paddle.device_count", None],
    "torch.manual_seed": ["paddle.seed", None],
    "torch.unsqueeze": ["paddle.unsqueeze", UnSqueezeMapper],
    "torch.squeeze": ["paddle.squeeze", UnSqueezeMapper],
    "torch.sum": ["x2paddle.torch2paddle.sum", None],
    "torch.mean": ["x2paddle.torch2paddle.mean", None],
    "torch.full": ["paddle.full", TensorBuilderMapper],
    "torch.full_like": ["paddle.full_like", TensorLikeMapper],
    "torch.ones": ["paddle.ones", TensorBuilderMapper],
    "torch.ones_like": ["paddle.full_like", TensorLikeMapper],
    "torch.zeros": ["paddle.zeros", TensorBuilderMapper],
    "torch.zeros_like": ["paddle.full_like", TensorLikeMapper],
    "torch.sqrt": ["paddle.sqrt", OneMathMapper],
    "torch.arange": ["paddle.arange", ArangeMapper],
    "torch.matmul": ["paddle.matmul", TwoMathMapper],
    "torch.set_grad_enabled": ["paddle.no_grad", NoGradMapper],
    "torch.tensor": ["paddle.to_tensor", None],
    "torch.clamp": ["paddle.clip", OneMathMapper],
    "torch.exp": ["paddle.exp", OneMathMapper],
    "torch.max": ["x2paddle.torch2paddle.max", None],
    "torch.min": ["x2paddle.torch2paddle.min", None],
    "torch.argmax": ["paddle.argmax", OneMathMapper],
    "torch.argmin": ["paddle.argmin", OneMathMapper],
    "torch.stack": ["paddle.stack", StackMapper],
    "torch.log": ["paddle.log", OneMathMapper],
    "torch.randperm": ["paddle.randperm", RandpermMapper],
    "torch.rand": ["x2paddle.torch2paddle.rand", None],
    "torch.randn_like": ["x2paddle.torch2paddle.randn_like", None],
    "torch.abs": ["paddle.abs", OneMathMapper],
    "torch.bitwise_or": ["paddle.logical_or", LogicalMapper],
    "torch.bitwise_xor": ["paddle.logical_xor", LogicalMapper],
    "torch.bitwise_and": ["paddle.logical_and", LogicalMapper],
    "torch.bitwise_not": ["paddle.logical_not", LogicalMapper],
    "torch.split": ["paddle.split", SplitMapper],
    "torch.hub.load_state_dict_from_url": ["paddle.load", HubLoadMapper],
    "torch.randn": ["x2paddle.torch2paddle.randn", None],
    "torch.add": ["paddle.add", TwoMathMapper],
    "torch.mul": ["paddle.multiply", TwoMathMapper],
    "torch.einsum": ["paddlenlp.ops.einsum ", None],
    "torch.linspace": ["paddle.linspace", LinspaceMapper],
    "torch.as_tensor": ["paddle.to_tensor", ToTensorMapper],
}
INVALID_API = {
    "torch.channels_last": ["None", None],
    "torch.cuda.empty_cache": ["x2paddle.torch2paddle.invalid", None],
}

API_MAPPER.update(OPTIMIZER_MAPPER)
API_MAPPER.update(NN_MAPPER)
API_MAPPER.update(UTILS_MAPPER)
API_MAPPER.update(DTYPE_MAPPER)
API_MAPPER.update(DIST_MAPPER)
API_MAPPER.update(TORCHVISION_MAPPER)
API_MAPPER.update(AUTOGRAD_MAPPER)
API_MAPPER.update(INVALID_API)

REMOVE_API = [
    "torch.backends.cudnn",
    "torch.backends.cudnn.benchmark",
    "torch.backends.cudnn.enabled",
    "torch.backends.cudnn.deterministic",
]
