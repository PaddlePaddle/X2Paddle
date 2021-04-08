from x2paddle.project_convertor.pytorch.api_mapper import *
from x2paddle.utils import *

OPTIMIZER_MAPPER = {
                    "torch.optim": 
                        ["paddle.optimizer", None],
                    "torch.optim.lr_scheduler.ReduceLROnPlateau":
                        ["paddle.optimizer.lr.ReduceOnPlateau", ClassLRScheculer],
                    "torch.optim.lr_scheduler.CosineAnnealingLR": 
                        ["paddle.optimizer.lr.CosineAnnealingDecay", ClassLRScheculer],
                    "torch.optim.lr_scheduler.MultiStepLR": 
                        ["paddle.optimizer.lr.MultiStepDecay", ClassLRScheculer],
                    "torch.optim.Adam": 
                        ["x2paddle.torch2paddle.Adam", None],
                    "torch.optim.SGD": 
                        ["x2paddle.torch2paddle.Momentum", None]
                   }

NN_MAPPER = {
             # basic
             "torch.nn": 
                 ["paddle.nn", None],
             "torch.nn.Module": 
                 ["paddle.nn.Layer", None],
             "torch.nn.ModuleList": 
                 ["paddle.nn.LayerList", None],
             "torch.nn.Sequential":
                 ["paddle.nn.Sequential", None],
             "torch.nn.utils": 
                 ["paddle.nn.utils", None],
             "torch.nn.utils.clip_grad_value_": 
                 ["x2paddle.torch2paddle.clip_grad_value_", None],
             "torch.nn.Parameter": 
                 ["paddle.create_parameter", FuncCreateParam],
             "torch.nn.DataParallel": 
                 ["paddle.DataParallel", ClassDataParallel],
             "torch.nn.functional": 
                 ["paddle.nn.functional", None],
             # nn_net
             "torch.nn.BatchNorm1d": 
                 ["paddle.nn.BatchNorm1D", ClassBatchNorm],
             "torch.nn.BatchNorm2d": 
                 ["paddle.nn.BatchNorm2D", ClassBatchNorm],
             "torch.nn.BatchNorm3d": 
                 ["paddle.nn.BatchNorm3D", ClassBatchNorm],
             "torch.nn.Conv1d": 
                 ["paddle.nn.Conv1D", ClassConv],
             "torch.nn.Conv2d": 
                 ["paddle.nn.Conv2D", ClassConv],
             "torch.nn.Conv3d": 
                 ["paddle.nn.Conv3D", ClassConv],
             "torch.nn.ConvTranspose2d": 
                 ["x2paddle.torch2paddle.Conv2DTranspose", None],
             "torch.nn.Dropout": 
                 ["paddle.nn.Dropout", ClassDropout],
             "torch.nn.Embedding": 
                 ["paddle.nn.Embedding", ClassEmbedding],
             "torch.nn.InstanceNorm2d": 
                 ["paddle.nn.InstanceNorm2D", ClassBatchNorm],
             "torch.nn.LeakyReLU": 
                 ["paddle.nn.LeakyReLU", None],
             "torch.nn.Linear": 
                 ["paddle.nn.Linear", ClassLinear],
             "torch.nn.MaxPool1d": 
                 ["paddle.nn.MaxPool1D", ClassMaxPool],
             "torch.nn.MaxPool2d": 
                 ["paddle.nn.MaxPool2D", ClassMaxPool],
             "torch.nn.MaxPool3d": 
                 ["paddle.nn.MaxPool3D", ClassMaxPool],
             "torch.nn.ReLU": 
                 ["paddle.nn.ReLU", ClassReLU],
             "torch.nn.Sigmoid": 
                 ["paddle.nn.Sigmoid", None],
             "torch.nn.Softmax": 
                 ["paddle.nn.Softmax", ClassSoftmax],
             "torch.nn.Tanh": 
                 ["paddle.nn.Tanh", None],
             "torch.nn.Upsample": 
                 ["paddle.nn.Upsample", None],
             # nn_loss
             "torch.nn.CrossEntropyLoss": 
                 ["paddle.nn.CrossEntropyLoss", ClassLoss],
             "torch.nn.BCEWithLogitsLoss": 
                 ["paddle.nn.BCEWithLogitsLoss", ClassLoss],
             "torch.nn.BCELoss": 
                 ["paddle.nn.BCELoss", None],
             # functional_net
            "torch.nn.functional.avg_pool1d": 
                 ["paddle.nn.functional.avg_pool1d", FuncAvgPool],
            "torch.nn.functional.avg_pool2d": 
                 ["paddle.nn.functional.avg_pool2d", FuncAvgPool],
            "torch.nn.functional.avg_pool3d": 
                 ["paddle.nn.functional.avg_pool3d", FuncAvgPool],
            "torch.nn.functional.dropout": 
                 ["paddle.nn.functional.dropout", FuncDropout],
             "torch.nn.functional.log_softmax": 
                 ["paddle.nn.functional.log_softmax", FuncLogSoftmax],
             "torch.nn.functional.pad": 
                 ["paddle.nn.functional.pad", FuncPad],
             "torch.sigmoid": 
                 ["paddle.nn.functional.sigmoid", FuncSigmoid],
             "torch.nn.functional.sigmoid":
                 ["paddle.nn.functional.sigmoid", FuncSigmoid],
             "torch.nn.functional.softmax": 
                 ["paddle.nn.functional.softmax", FuncSoftmax],
             "torch.nn.init.xavier_uniform_": 
                 ["x2paddle.torch2paddle.xavier_uniform_", FuncXavierUniform],
            # functional_loss
             "torch.nn.functional.binary_cross_entropy_with_logits": 
                 ["x2paddle.torch2paddle.binary_cross_entropy_with_logits", None],
             "torch.nn.functional.cross_entropy": 
                 ["paddle.nn.functional.cross_entropy", FuncCrossEntropy],
             "torch.nn.functional.dropout":
                 ["paddle.nn.functional.dropout", FuncDropout],
             "torch.nn.functional.relu": 
                 ["paddle.nn.functional.relu", FuncRelu],
             "torch.nn.functional.smooth_l1_loss": 
                 ["paddle.nn.functional.smooth_l1_loss", FuncSmoothL1Loss],}

UTILS_MAPPER = {
                "torch.utils.data": 
                    ["paddle.io", None],
                "torch.utils.data.DataLoader": 
                    ["x2paddle.torch2paddle.DataLoader", None], 
                "torch.utils.data.random_split": 
                    ["x2paddle.torch2paddle.random_split", None],
                "torch.utils.data.Dataset": 
                    ["paddle.io.Dataset", None],
                "torch.utils.data.ConcatDataset": 
                    ["x2paddle.torch2paddle.ConcatDataset", None]
               }

DTYPE_MAPPER = {
                "torch.float32": 
                    [string("float32"), None],
                "torch.long": 
                    [string("int64"), None]
                }

TORCHVISION_MAPPER  = {
                       "torchvision.transforms": 
                           ["paddle.vision.transforms", None],
                       "torchvision.transforms.Compose": 
                           ["paddle.vision.transforms.Compose", None],
                       "torchvision.transforms.ToPILImage": 
                           ["x2paddle.torch2paddle.ToPILImage", None],
                       "torchvision.transforms.Resize": 
                           ["paddle.vision.transforms.Resize", None],
                       "torchvision.transforms.ToTensor": 
                           ["x2paddle.torch2paddle.ToTensor", None],
                       "torchvision.transforms.RandomHorizontalFlip": 
                           ["paddle.vision.transforms.RandomHorizontalFlip", None],
                       "torchvision.transforms.CenterCrop": 
                           ["paddle.vision.transforms.CenterCrop", None],
                       "torchvision.transforms.Normalize": 
                           ["x2paddle.torch2paddle.Normalize", None],
                       "torchvision.utils.save_image": 
                           ["x2paddle.torch2paddle.save_image", None],
                       "torchvision.datasets.ImageFolder": 
                           ["paddle.vision.datasets.ImageFolder", ClassImageFolder]
                        }

AUTOGRAD_MAPPER = {
                   "torch.autograd.Variable": ["paddle.to_tensor", None], # TODO(syf): 确认是否一致
                   "torch.autograd.grad": ["paddle.grad", None],
                  }

API_MAPPER = {
              "torch": 
                  ["paddle", None],
              "torch.Tensor": 
                  ["paddle.Tensor", None],
              "torch.FloatTensor": 
                  ["paddle.Tensor", ClassFloatTensor],
              "torch.load": 
                  ["paddle.load", FuncLoad],
              "torch.save": 
                  ["paddle.save", FuncSave],
              "torch.device": 
                  ["paddle.set_device", FuncSetDevice],
              "torch.cat": 
                  ["x2paddle.torch2paddle.concat", None],
              "torch.cuda.is_available": 
                  ["paddle.is_compiled_with_cuda", None],
              "torch.no_grad": 
                  ["paddle.no_grad", None],
              "torch.from_numpy": 
                  ["paddle.to_tensor", None],
              "torch.cuda.device_count": 
                  ["x2paddle.torch2paddle.device_count", None],
              "torch.manual_seed": 
                  ["paddle.seed", None],
              "torch.unsqueeze": 
                  ["paddle.unsqueeze", FuncUnSqueeze],
              "torch.squeeze": 
                  ["paddle.squeeze", FuncUnSqueeze],
              "torch.sum": 
                  ["x2paddle.torch2paddle.sum", None],
              "torch.mean": 
                  ["x2paddle.torch2paddle.mean", None],
              "torch.full": 
                  ["paddle.full", FunTensorBuilder],
              "torch.full_like": 
                  ["paddle.full_like", FunTensorLike],
              "torch.ones": 
                  ["paddle.ones", FunTensorBuilder],
              "torch.ones_like": 
                  ["paddle.full_like", FunTensorLike],
              "torch.zeros": 
                  ["paddle.zeros", FunTensorBuilder],
              "torch.zeros_like": 
                  ["paddle.full_like", FunTensorLike],
              "torch.sqrt": 
                  ["paddle.sqrt", FuncMath],
              "torch.arange": 
                  ["paddle.arange", FuncArange],
              "torch.matmul": 
                  ["paddle.matmul", FuncMatmul],
              "torch.set_grad_enabled": 
                  ["paddle.no_grad", FuncNoGrad],
              "torch.tensor": 
                  ["paddle.to_tensor", None],
              "torch.clamp": 
                  ["paddle.clip", FuncMath],
              "torch.exp": 
                  ["paddle.exp", FuncMath],
              "torch.max": 
                  ["x2paddle.torch2paddle.max", None],
              "torch.min": 
                  ["x2paddle.torch2paddle.min", None],
              "torch.argmax": 
                  ["paddle.argmax", FuncMath],
              "torch.argmin": 
                  ["paddle.argmin", FuncMath],
              "torch.stack": 
                  ["paddle.stacks", FuncStack],
              "torch.log": 
                  ["paddle.log", FuncMath],
              "torch.randperm": 
                  ["paddle.randperm", FuncRandperm],
              "torch.rand": 
                  ["x2paddle.torch2paddle.rand", None],
              "torch.abs": 
                  ["paddle.abs", FuncMath],
              "torch.bitwise_or": 
                  ["paddle.logical_or", FuncLogical],
              "torch.bitwise_xor": 
                  ["paddle.logical_xor", FuncLogical],
              "torch.bitwise_and": 
                  ["paddle.logical_and", FuncLogical],
              "torch.bitwise_not": 
                  ["paddle.logical_not", FuncLogical],
             }

API_MAPPER.update(OPTIMIZER_MAPPER)
API_MAPPER.update(NN_MAPPER)
API_MAPPER.update(UTILS_MAPPER) 
API_MAPPER.update(DTYPE_MAPPER) 
API_MAPPER.update(TORCHVISION_MAPPER)
API_MAPPER.update(AUTOGRAD_MAPPER)

REMOVE_API =["torch.backends.cudnn",
             "torch.backends.cudnn.benchmark"]
