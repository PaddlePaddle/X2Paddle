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

from .utils import *
from x2paddle.utils import *


class BatchNormModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "num_features", "num_channels")
        rename_key(self.kwargs, "eps", "epsilon")
        rename_key(self.kwargs, "track_running_stats", "use_global_stats")
        if "momentum" in self.kwargs:
            if isinstance(self.kwargs["momentum"], float):
                self.kwargs["momentum"] = 1 - self.kwargs["momentum"]
            else:
                self.kwargs["momentum"] = "1 - {}".format(self.kwargs[
                    "momentum"])
        if "affine" in self.kwargs and not self.kwargs["affine"]:
            for key in ["weight_attr", "bias_attr"]:
                self.kwargs[key] = "paddle.ParamAttr(learning_rate=0.0)"

    def delete_attrs(self):
        delete_key(self.kwargs, "affine")

    def run(self):
        if self.pytorch_api_name == "torch.nn.BatchNorm1d" and self.rename_func_name(
                "x2paddle.torch2paddle.BatchNorm1D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.BatchNorm2d" and self.rename_func_name(
                "x2paddle.torch2paddle.BatchNorm2D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.BatchNorm3d" and self.rename_func_name(
                "x2paddle.torch2paddle.BatchNorm3D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.InstanceNorm2d" and self.rename_func_name(
                "x2paddle.torch2paddle.InstanceNorm2D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            if "affine" in self.kwargs and not (isinstance(self.kwargs["affine"], bool) \
                           or (isinstance(self.kwargs["affine"], str) and self.kwargs["affine"].strip() in ["True", "False"])):
                print(self.kwargs["affine"], self.func_name)
                if self.pytorch_api_name == "torch.nn.BatchNorm1D":
                    self.func_name = "x2paddle.torch2paddle.BatchNorm1D"
                elif self.pytorch_api_name == "torch.nn.BatchNorm2D":
                    self.func_name = "x2paddle.torch2paddle.BatchNorm2D"
                elif self.pytorch_api_name == "torch.nn.BatchNorm3D":
                    self.func_name = "x2paddle.torch2paddle.BatchNorm3D"
                elif self.pytorch_api_name == "torch.nn.InstanceNorm2D":
                    self.func_name = "x2paddle.torch2paddle.InstanceNorm2D"
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
            return self.convert_to_paddle()


class ConvModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "bias", "bias_attr")

    def run(self):
        if self.pytorch_api_name == "torch.nn.Conv1d" and self.rename_func_name(
                "x2paddle.torch2paddle.Conv1D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.Conv2d" and self.rename_func_name(
                "x2paddle.torch2paddle.Conv2D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.Conv3d" and self.rename_func_name(
                "x2paddle.torch2paddle.Conv3D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(7)
            return self.convert_to_paddle()


class DropoutModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")


class EmbeddingModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "max_norm")
        delete_key(self.kwargs, "norm_type")
        delete_key(self.kwargs, "scale_grad_by_freq")

    def check_attrs(self):
        assert "max_norm" not in self.kwargs or self.kwargs[
            "max_norm"] is None, "The max_norm is not supported yet in Embedding!"
        assert "scale_grad_by_freq" not in self.kwargs or not self.kwargs[
            "scale_grad_by_freq"], "The scale_grad_by_freq must be False in Embedding!"

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.Embedding"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(3)
            return self.convert_to_paddle()


class LinearModuleMapper(ConvModuleMapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.Linear"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()


class LossModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "size_average")
        delete_key(self.kwargs, "reduce")

    def process_attrs(self):
        rename_key(self.kwargs, "target", "label")

    def run(self):
        if self.pytorch_api_name == "torch.nn.CrossEntropyLoss" and \
        self.rename_func_name("x2paddle.torch2paddle.CrossEntropyLoss"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.BCEWithLogitsLoss" and \
        self.rename_func_name("x2paddle.torch2paddle.BCEWithLogitsLoss"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_paddle()


class MaxPoolModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "return_indices", "return_mask")

    def check_attrs(self):
        assert "dilation" not in self.kwargs, "The dilation is not supported yet in MaxPool!"

    def run(self):
        if self.pytorch_api_name == "torch.nn.MaxPool1d" and self.rename_func_name(
                "x2paddle.torch2paddle.MaxPool1D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.MaxPool2d" and self.rename_func_name(
                "x2paddle.torch2paddle.MaxPool12D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.MaxPool3d" and self.rename_func_name(
                "x2paddle.torch2paddle.MaxPool13D"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(3)
            return self.convert_to_paddle()


class ReLUModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")
        if len(self.args) > 0:
            self.args.clear()


class SoftmaxModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "dim", "axis")


class AvgPoolFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        if "count_include_pad" in self.kwargs:
            if isinstance(self.kwargs["count_include_pad"], bool):
                self.kwargs["exclusive"] = not self.kwargs["count_include_pad"]
            else:
                self.kwargs["exclusive"] = "not {}".format(self.kwargs[
                    "count_include_pad"])

    def delete_attrs(self):
        delete_key(self.kwargs, "count_include_pad")

    def run(self):
        if self.pytorch_api_name == "torch.nn.functional.avg_pool1d" and self.rename_func_name(
                "x2paddle.torch2paddle.avg_pool1d"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.functional.avg_pool2d" and self.rename_func_name(
                "x2paddle.torch2paddle.avg_pool2d"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        elif self.pytorch_api_name == "torch.nn.functional.avg_pool3d" and self.rename_func_name(
                "x2paddle.torch2paddle.avg_pool3d"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(4)
            return self.convert_to_paddle()


class CrossEntropyFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "target", "label")


class DropoutFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")

    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.dropout"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()


class LogSoftmaxFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "dim", "axis")
        rename_key(self.kwargs, "input", "x")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.log_softmax"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()


class PadFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")


class ReluFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")

    def delete_attrs(self):
        delete_key(self.kwargs, "inplace")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.relu"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            return self.convert_to_paddle()


class SigmoidFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")


class SmoothL1LossFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "beta", "delta")
        rename_key(self.kwargs, "target", "label")

    def delete_attrs(self):
        delete_key(self.kwargs, "size_average")
        delete_key(self.kwargs, "reduce")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.smooth_l1_loss"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()


class SoftmaxFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "dim", "axis")

    def delete_attrs(self):
        delete_key(self.kwargs, "_stacklevel")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.softmax"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()


class XavierUniformFuncMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def run(self):
        if len(self.args) >= 1:
            param_name = self.args[0]
            if len(self.args) > 1:
                gain = self.args[1]
            else:
                gain = self.kwargs["gain"] if "gain" in self.kwargs else 1.0
        else:
            param_name = self.kwargs["tensor"]
            gain = self.kwargs["gain"] if "gain" in self.kwargs else 1.0
        param_name = param_name.strip()
        code = "{} = {}(gain={})".format(param_name, self.func_name, gain)
        return [], code, []
