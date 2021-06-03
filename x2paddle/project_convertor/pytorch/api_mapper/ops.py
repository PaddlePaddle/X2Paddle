from .utils import *
from x2paddle.utils import *


class SaveMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "pickle_module")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.save"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()


class LoadMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "pickle_module")
        delete_key(self.kwargs, "map_location")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.load"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()


class HubLoadMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        if len(self.args) == 1:
            self.kwargs.clear()
        elif len(self.args) == 0:
            self.args.append(list(self.kwargs.values())[0])
            self.kwargs.clear()
        else:
            self.args = self.args[:1]

    def run(self):
        if self.pytorch_api_name == "torch.hub.load_state_dict_from_url":
            if self.rename_func_name(
                    "x2paddle.torch2paddle.load_state_dict_from_url"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.utils.model_zoo.load_url":
            if self.rename_func_name("x2paddle.torch2paddle.load_url"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        return self.convert_to_paddle()


class SetDeviceMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()

    def process_attrs(self):
        self.useful_attrs["device"] = self.args[0]
        self.args[0] = self.target_name

    def run(self):
        self.process_attrs()
        insert_codes = list()
        if self.target_name is None:
            self.target_name = "tmp"
        insert_codes.append("{} = {}".format(self.target_name,
                                             self.useful_attrs["device"]))
        insert_codes.append("{} = {}.replace('cuda', 'gpu')".format(
            self.target_name, self.target_name))
        return insert_codes, generate_api_code(self.func_name, self.args,
                                               self.kwargs), []


class DataParallelModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "device_ids")
        delete_key(self.kwargs, "output_device")
        delete_key(self.kwargs, "dim")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.DataParallel"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            return self.convert_to_paddle()


class UnSqueezeMapper(Mapper):
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

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.unsqueeze"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(2)
            return self.convert_to_paddle()


class OneMathMapper(Mapper):
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
        delete_key(self.kwargs, "out")

    def run(self):
        if self.pytorch_api_name == "torch.sqrt":
            if self.rename_func_name("x2paddle.torch2paddle.sqrt"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.abs":
            if self.rename_func_name("x2paddle.torch2paddle.abs"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.log":
            if self.rename_func_name("x2paddle.torch2paddle.log"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.exp":
            if self.rename_func_name("x2paddle.torch2paddle.exp"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.clip":
            if self.rename_func_name("x2paddle.torch2paddle.clip"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        return self.convert_to_paddle()


class ArangeMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()

    def process_attrs(self):
        self.useful_attrs["requires_grad"] = self.kwargs[
            "requires_grad"] if "requires_grad" in self.kwargs else False

    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.arange"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(3)
            out1, out2, out3 = self.convert_to_paddle()
            if isinstance(self.useful_attrs["requires_grad"],
                          str) or not self.useful_attrs["requires_grad"]:
                out2 = "{}.requires_grad_({})".format(
                    out2, self.useful_attrs["requires_grad"])
            return out1, out2, out3


class TwoMathMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")

    def delete_attrs(self):
        delete_key(self.kwargs, "out")

    def run(self):
        if self.pytorch_api_name == "torch.matmul":
            if self.rename_func_name("x2paddle.torch2paddle.matmul"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.add":
            if self.rename_func_name("x2paddle.torch2paddle.add"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.mul":
            if self.rename_func_name("x2paddle.torch2paddle.mul"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        return self.convert_to_paddle()


class CreateParamModuleMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "data", "value")

    def delete_attrs(self):
        delete_key(self.kwargs, "requires_grad")

    def check_attrs(self):
        assert "requires_grad" not in self.kwargs or self.kwargs[
            "requires_grad"], "The requires_grad must be True in Parameter!"

    def run(self):
        if self.rename_func_name(self.func_name):
            if "*" in self.args[0] and "**" not in self.args[0]:
                param_name = self.args[0][1:]
            elif "**" in self.args[0]:
                param_name = "{}['data']".format(self.args[0][2:])
            elif "*" not in self.args[0] and "**" in self.args[1]:
                param_name = self.args[0]
        else:
            self.check_attrs()
            self.process_attrs()
            self.delete_attrs()
            if len(self.args) == 1:
                param_name = self.args[0]
            else:
                param_name = self.kwargs["value"]
        code = "paddle.create_parameter(shape={}.shape, dtype=str({}.numpy().dtype), default_initializer = paddle.nn.initializer.Assign({}))".format(
            param_name, param_name, param_name)
        return [], code, ["{}.stop_gradient = False".format(self.target_name)]


class NoGradMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        self.args.clear()
        self.kwargs.clear()


class LogicalMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")

    def run(self):
        if self.pytorch_api_name == "torch.bitwise_or":
            if self.rename_func_name("x2paddle.torch2paddle.logical_or"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.bitwise_and":
            if self.rename_func_name("x2paddle.torch2paddle.logical_and"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.bitwise_xor":
            if self.rename_func_name("x2paddle.torch2paddle.logical_xor"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.bitwise_not":
            if self.rename_func_name("x2paddle.torch2paddle.logical_not"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        return self.convert_to_paddle()


class StackMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "tensors", "x")
        rename_key(self.kwargs, "dim", "axis")

    def delete_attrs(self):
        delete_key(self.kwargs, "out")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.stack"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            return self.convert_to_paddle()


class RandpermMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")
        delete_key(self.kwargs, "pin_memory")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.randperm"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            return self.convert_to_paddle()


class TensorBuilderMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()

    def process_attrs(self):
        if self.pytorch_api_name in ["torch.ones", "torch.zeros"]:
            if len(self.args) > 1:
                new_args = list()
                for arg in self.args:
                    if isinstance(arg, int):
                        new_args.append(str(arg))
                    else:
                        new_args.append(arg)
                shape = ", ".join(new_args)
                shape = "[{}]".format(shape)
                self.args.clear()
                self.args.append(shape)
        rename_key(self.kwargs, "size", "shape")
        self.useful_attrs["requires_grad"] = self.kwargs[
            "requires_grad"] if "requires_grad" in self.kwargs else False

    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")

    def run(self):
        if self.pytorch_api_name == "torch.full":
            if self.rename_func_name("x2paddle.torch2paddle.full"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.zeros":
            if self.rename_func_name("x2paddle.torch2paddle.zeros"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.ones":
            if self.rename_func_name("x2paddle.torch2paddle.ones"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        out1, out2, out3 = self.convert_to_paddle()
        if isinstance(self.useful_attrs["requires_grad"],
                      str) or not self.useful_attrs["requires_grad"]:
            out2 = "{}.requires_grad_({})".format(
                out2, self.useful_attrs["requires_grad"])
        return out1, out2, out3


class TensorLikeMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()

    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        self.useful_attrs["requires_grad"] = self.kwargs[
            "requires_grad"] if "requires_grad" in self.kwargs else False

    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")

    def run(self):
        if self.pytorch_api_name == "torch.full_like":
            if self.rename_func_name("x2paddle.torch2paddle.full_like"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.zeros_like":
            if self.rename_func_name("x2paddle.torch2paddle.zeros_like"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        if self.pytorch_api_name == "torch.ones_like":
            if self.rename_func_name("x2paddle.torch2paddle.ones_like"):
                return [], generate_api_code(self.func_name, self.args,
                                             self.kwargs), []
        out1, out2, out3 = self.convert_to_paddle()
        if isinstance(self.useful_attrs["requires_grad"],
                      str) or not self.useful_attrs["requires_grad"]:
            out2 = "{}.requires_grad_({})".format(
                out2, self.useful_attrs["requires_grad"])
        return out1, out2, out3


class SplitMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "tensor", "x")
        rename_key(self.kwargs, "split_size_or_sections", "num_or_sections")
        rename_key(self.kwargs, "dim", "axis")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.split"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(3)
            return self.convert_to_paddle()


class LinspaceMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        self.useful_attrs = dict()

    def process_attrs(self):
        rename_key(self.kwargs, "end", "stop")
        rename_key(self.kwargs, "steps", "num")
        self.useful_attrs["requires_grad"] = self.kwargs[
            "requires_grad"] if "requires_grad" in self.kwargs else False

    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        delete_key(self.kwargs, "layout")
        delete_key(self.kwargs, "device")
        delete_key(self.kwargs, "requires_grad")

    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.linspace"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs(1)
            out1, out2, out3 = self.convert_to_paddle()
            if isinstance(self.useful_attrs["requires_grad"],
                          str) or not self.useful_attrs["requires_grad"]:
                out2 = "{}.requires_grad_({})".format(
                    out2, self.useful_attrs["requires_grad"])
            return out1, out2, out3


class ToTensorMapper(Mapper):
    def __init__(self,
                 func_name,
                 pytorch_api_name,
                 args,
                 kwargs,
                 target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)

    def process_attrs(self):
        rename_key(self.kwargs, "device", "place")

    def run(self):
        if self.rename_func_name("paddle.to_tensor"):
            return [], generate_api_code(self.func_name, self.args,
                                         self.kwargs), []
        else:
            self.convert_args2kwargs()
            return self.convert_to_paddle()
