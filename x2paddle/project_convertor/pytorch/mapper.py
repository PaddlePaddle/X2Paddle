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
from x2paddle.utils import *

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
    "torch.nn.Conv1d": ["paddle.nn.Conv1D", ConvModuleMapper],
    "torch.nn.Conv2d": ["paddle.nn.Conv2D", ConvModuleMapper],
    "torch.nn.Conv3d": ["paddle.nn.Conv3D", ConvModuleMapper],
    "torch.nn.ConvTranspose2d":
    ["x2paddle.torch2paddle.Conv2DTranspose", None],
    "torch.nn.Dropout": ["paddle.nn.Dropout", DropoutModuleMapper],
    "torch.nn.Embedding": ["paddle.nn.Embedding", EmbeddingModuleMapper],
    "torch.nn.InstanceNorm2d":
    ["paddle.nn.InstanceNorm2D", BatchNormModuleMapper],
    "torch.nn.LeakyReLU": ["paddle.nn.LeakyReLU", None],
    "torch.nn.Linear": ["paddle.nn.Linear", LinearModuleMapper],
    "torch.nn.MaxPool1d": ["paddle.nn.MaxPool1D", MaxPoolModuleMapper],
    "torch.nn.MaxPool2d": ["paddle.nn.MaxPool2D", MaxPoolModuleMapper],
    "torch.nn.MaxPool3d": ["paddle.nn.MaxPool3D", MaxPoolModuleMapper],
    "torch.nn.ReLU": ["paddle.nn.ReLU", ReLUModuleMapper],
    "torch.nn.ReLU6": ["paddle.nn.ReLU6", ReLUModuleMapper],
    "torch.nn.Sigmoid": ["paddle.nn.Sigmoid", None],
    "torch.nn.Softmax": ["paddle.nn.Softmax", SoftmaxModuleMapper],
    "torch.nn.Tanh": ["paddle.nn.Tanh", None],
    "torch.nn.Upsample": ["paddle.nn.Upsample", None],
    # nn_loss
    "torch.nn.CrossEntropyLoss":
    ["paddle.nn.CrossEntropyLoss", LossModuleMapper],
    "torch.nn.BCEWithLogitsLoss":
    ["paddle.nn.BCEWithLogitsLoss", LossModuleMapper],
    "torch.nn.BCELoss": ["paddle.nn.BCELoss", None],
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
    "torch.nn.functional.log_softmax":
    ["paddle.nn.functional.log_softmax", LogSoftmaxFuncMapper],
    "torch.nn.functional.pad": ["paddle.nn.functional.pad", PadFuncMapper],
    "torch.sigmoid": ["paddle.nn.functional.sigmoid", SigmoidFuncMapper],
    "torch.nn.functional.sigmoid":
    ["paddle.nn.functional.sigmoid", SigmoidFuncMapper],
    "torch.nn.functional.softmax":
    ["paddle.nn.functional.softmax", SoftmaxFuncMapper],
    # init
    "torch.nn.init": ["x2paddle.torch2paddle", None],
    "torch.nn.init.kaiming_normal_":
    ["x2paddle.torch2paddle.kaiming_normal_", None],
    "torch.nn.init.xavier_uniform_":
    ["x2paddle.torch2paddle.xavier_normal_", None],
    "torch.nn.init.xavier_normal_":
    ["x2paddle.torch2paddle.xavier_uniform_", None],
    "torch.nn.init.ones_": ["x2paddle.torch2paddle.ones_init_", None],
    "torch.nn.init.zeros_": ["x2paddle.torch2paddle.zeros_init_", None],
    # functional_loss
    "torch.nn.functional.binary_cross_entropy_with_logits":
    ["x2paddle.torch2paddle.binary_cross_entropy_with_logits", None],
    "torch.nn.functional.cross_entropy":
    ["paddle.nn.functional.cross_entropy", CrossEntropyFuncMapper],
    "torch.nn.functional.dropout":
    ["paddle.nn.functional.dropout", DropoutFuncMapper],
    "torch.nn.functional.relu": ["paddle.nn.functional.relu", ReluFuncMapper],
    "torch.nn.functional.smooth_l1_loss":
    ["paddle.nn.functional.smooth_l1_loss", SmoothL1LossFuncMapper],
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

DTYPE_MAPPER = {
    "torch.float32": [string("float32"), None],
    "torch.long": [string("int64"), None]
}

TORCHVISION_MAPPER = {
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
    "torchvision.utils": ["x2paddle.torch2paddle", None],
    "torchvision.utils.save_image": ["x2paddle.torch2paddle.save_image", None],
    "torchvision.datasets": ["paddle.vision.datasets", None],
    "torchvision.datasets.ImageFolder":
    ["x2paddle.torch2paddle.ImageFolder", None]
}

AUTOGRAD_MAPPER = {
    "torch.autograd.Variable": ["paddle.to_tensor", None],  # TODO(syf): 确认是否一致
    "torch.autograd.grad": ["paddle.grad", None],
}

API_MAPPER = {
    "torch": ["paddle", None],
    "torch.Tensor": ["x2paddle.torch2paddle.create_tensor", None],
    "torch.FloatTensor": ["paddle.Tensor", FloatTensorMapper],
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
    "torch.sqrt": ["paddle.sqrt", MathMapper],
    "torch.arange": ["paddle.arange", ArangeMapper],
    "torch.matmul": ["paddle.matmul", MatmulMapper],
    "torch.set_grad_enabled": ["paddle.no_grad", NoGradMapper],
    "torch.tensor": ["paddle.to_tensor", None],
    "torch.clamp": ["paddle.clip", MathMapper],
    "torch.exp": ["paddle.exp", MathMapper],
    "torch.max": ["x2paddle.torch2paddle.max", None],
    "torch.min": ["x2paddle.torch2paddle.min", None],
    "torch.argmax": ["paddle.argmax", MathMapper],
    "torch.argmin": ["paddle.argmin", MathMapper],
    "torch.stack": ["paddle.stacks", StackMapper],
    "torch.log": ["paddle.log", MathMapper],
    "torch.randperm": ["paddle.randperm", RandpermMapper],
    "torch.rand": ["x2paddle.torch2paddle.rand", None],
    "torch.abs": ["paddle.abs", MathMapper],
    "torch.bitwise_or": ["paddle.logical_or", LogicalMapper],
    "torch.bitwise_xor": ["paddle.logical_xor", LogicalMapper],
    "torch.bitwise_and": ["paddle.logical_and", LogicalMapper],
    "torch.bitwise_not": ["paddle.logical_not", LogicalMapper],
    "torch.split": ["paddle.split", SplitMapper],
    "torch.hub.load_state_dict_from_url": ["paddle.load", HubLoadMapper],
}

API_MAPPER.update(OPTIMIZER_MAPPER)
API_MAPPER.update(NN_MAPPER)
API_MAPPER.update(UTILS_MAPPER)
API_MAPPER.update(DTYPE_MAPPER)
API_MAPPER.update(DIST_MAPPER)
API_MAPPER.update(TORCHVISION_MAPPER)
API_MAPPER.update(AUTOGRAD_MAPPER)

REMOVE_API = [
    "torch.backends.cudnn", "torch.backends.cudnn.benchmark",
    "torch.backends.cudnn.enabled", "torch.backends.cudnn.deterministic"
]
