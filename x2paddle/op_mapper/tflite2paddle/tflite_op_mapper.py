#   Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import sys
import numpy as np
from itertools import chain
from x2paddle.core.util import *
from x2paddle.core.program import PaddleGraph
from tflite.BuiltinOptions import BuiltinOptions
from tflite.TensorType import TensorType


def build_str_map(obj):
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith("_"):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret


def rename(name):
    name = name.replace("/", "_")
    return name


class TensorWrapper(object):
    """用于记录TFLite Tensor相关信息。"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params


class TFLiteOpMapper():
    def __init__(self, decoder):
        try:
            from tflite.BuiltinOperator import BuiltinOperator
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")
        self.model = decoder.model
        self.graph = decoder.graph
        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())
        if not self.op_checker():
            raise Exception("Model is not supported yet.")
        self.params = dict()
        self.paddle_graph = PaddleGraph(parent_layer=None, source_type="tflite")
        print("Total nodes: {}".format(self.graph.OperatorsLength()))
        self.inputs_info = {}
        self.nn_name2id = {}
        print("Nodes converting ...")
        for ipt_name in decoder.inputs_name:
            self.convert_input(ipt_name, decoder.shape_dict[ipt_name],
                               decoder.dtype_dict[ipt_name])

        for op_idx in range(self.graph.OperatorsLength()):
            sys.stderr.write("\rConverting node {} ...     ".format(op_idx + 1))
            op = self.graph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            func = getattr(self, op_code_str)
            func(op)
        print("\nNodes converted.")
        self.paddle_graph.set_parameters(self.params)
        self.paddle_graph.set_inputs_info(self.inputs_info)
        self.paddle_graph.set_name("TFLiteModel")

    def op_checker(self):
        unsupported_ops = set()
        for op_idx in range(self.graph.OperatorsLength()):
            op = self.graph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            if not hasattr(self, op_code_str):
                unsupported_ops.add(op_code_str)
        if len(unsupported_ops) == 0:
            return True
        else:
            if len(unsupported_ops) > 0:
                print("\n========= {} OPs are not supported yet ===========".
                      format(len(unsupported_ops)))
            for op_code_str in unsupported_ops:
                print("========== {} ============".format(op_code_str))
            return False

    def get_op_code_str(self, op):
        """获取TFLite op的名字。"""
        try:
            from tflite.BuiltinOperator import BuiltinOperator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        op_code_list_idx = op.OpcodeIndex()
        op_code_id = self.model.OperatorCodes(op_code_list_idx).BuiltinCode()
        try:
            op_code_str = self.builtin_op_code[op_code_id]
        except KeyError:
            raise NotImplementedError(
                "TFLite operator with code " + str(op_code_id) +
                " is not supported by this version of the fbs schema.")
        if op_code_id == BuiltinOperator.CUSTOM:
            # Custom operator
            custom_op_code_str = self.model.OperatorCodes(
                op_code_list_idx).CustomCode()
            if custom_op_code_str == b"TFLite_Detection_PostProcess":
                return "DETECTION_POSTPROCESS"

            raise NotImplementedError(
                "Custom operators are currently not supported")
        return op_code_str

    def get_input_tensors(self, op):
        operator_outputs = op.InputsAsNumpy()
        return self.get_tensors(operator_outputs)

    def get_output_tensors(self, op):
        operator_inputs = op.OutputsAsNumpy()
        return self.get_tensors(operator_inputs)

    def get_tensors(self, tensors_idx_list):
        """Get tensor wrapper list from given TFLite tensor index list"""
        return_list = list()
        for tensor_idx in tensors_idx_list:
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, 0, 0))
                continue

            tensor = self.graph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                tflite_scale = tflite_qnn_params.ScaleAsNumpy()
                tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
                is_qnn_params_valid = True

                # Handle Per-axis and per-tensor cases
                if isinstance(tflite_scale, np.ndarray):
                    assert isinstance(tflite_zero_point, np.ndarray)

                    # Tensor - Per-axis quantization
                    if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                        scale = tflite_scale
                        # Ensure that all zero points are zeros
                        zero_point = tflite_zero_point
                        if not np.all(zero_point == 0):
                            raise Error(
                                "TFLite per-axis quantization restricts all zero points to be"
                                + " 0, but a non-zero value is observed")
                        zero_point = int(zero_point[0])

                    # Scalar - Per-tensor quantization
                    elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                        scale = float(tflite_scale[0])
                        zero_point = int(tflite_zero_point[0])

                    else:
                        raise NotImplementedError(
                            "Quantized type {} (scale) and  {} (zero point) not supported".
                            format(
                                type(tflite_scale), type(tflite_zero_point)))
                elif tflite_scale == 0 and tflite_zero_point == 0:
                    # Handle corner case for ops like quantized reshape whose second operand (shape)
                    # has zero scale and zero zero point. This is not used.
                    is_qnn_params_valid = False
                else:
                    raise NotImplementedError("Quantized type {} not supported".
                                              format(type(tflite_scale)))

                # Check that the scale and zero points are valid.
                if is_qnn_params_valid:
                    qnn_params = dict()
                    qnn_params["scale"] = scale
                    qnn_params["zero_point"] = zero_point
            return_list.append(
                TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
        return return_list

    def get_tensor_type_as_numpy(self, tensor_wrapper):
        """Returns np.dtype out of TensorType"""
        assert isinstance(tensor_wrapper, TensorWrapper)
        try:
            return {
                TensorType.UINT8: np.uint8,
                TensorType.INT8: np.int8,
                TensorType.FLOAT16: np.float16,
                TensorType.FLOAT32: np.float32,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.BOOL: np.bool_,
            }[tensor_wrapper.tensor.Type()]
        except KeyError:
            raise NotImplementedError(
                "Tensor type '{}' currently not supported".format(
                    tensor_wrapper.tensor.Type()))

    def get_tensor_value(self, tensor_wrapper):
        """Get tensor buffer value from given tensor wrapper"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        dtype = self.get_tensor_type_as_numpy(tensor_wrapper)
        data = tensor_wrapper.buffer.DataAsNumpy()

        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = tensor_wrapper.tensor.ShapeAsNumpy()
        else:
            shape = []

        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def convert_fused_activation_function(self, fused_activation_fn, name):
        try:
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if fused_activation_fn == ActivationFunctionType.NONE:
            return
        elif fused_activation_fn == ActivationFunctionType.RELU6:
            relu_name = name_generator("relu6", self.nn_name2id)
            layer_outputs = [relu_name, name]
            self.paddle_graph.add_layer(
                "paddle.nn.ReLU6",
                inputs={"input": name},
                outputs=layer_outputs)
        elif fused_activation_fn == ActivationFunctionType.RELU:
            relu_name = name_generator("relu", self.nn_name2id)
            layer_outputs = [relu_name, name]
            self.paddle_graph.add_layer(
                "paddle.nn.ReLU", inputs={"input": name}, outputs=layer_outputs)
        elif fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
            self.paddle_graph.add_layer(
                "paddle.clip",
                inputs={"x": name},
                outputs=layer_outputs,
                min=-1,
                max=1)
        elif fused_activation_fn == ActivationFunctionType.TANH:
            tanh_name = name_generator("tanh", self.nn_name2id)
            layer_outputs = [tanh_name, name]
            self.paddle_graph.add_layer(
                "paddle.nn.Tanh", inputs={"input": name}, outputs=layer_outputs)
        else:
            fused_activation_fn_str = self.activation_fn_type[
                fused_activation_fn]
            raise NotImplementedError(
                "Fused activation {} is not supported yet.".format(
                    fused_activation_fn_str))

    def convert_input(self, name, shape, dtype):
        name = rename(name)
        self.paddle_graph.add_layer(
            kernel="paddle.to_tensor", inputs={}, outputs=[name], data=name)
        self.inputs_info[name] = [shape, dtype]

    def convert_pool2d(self, op, pool_type):
        from tflite.Pool2DOptions import Pool2DOptions
        from tflite.Padding import Padding

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "There should be only 1 input tensor"
        input_tensor = input_tensors[0]
        input_tensor_name = input_tensor.tensor.Name().decode("utf8")
        input_tensor_name = rename(input_tensor_name)
        _, input_h, input_w, _ = input_tensor.tensor.ShapeAsNumpy()

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors should be 1"
        output_tensor = output_tensors[0]
        output_tensor_name = output_tensor.tensor.Name().decode("utf8")
        output_tensor_name = rename(output_tensor_name)

        pool2d_name = name_generator("pool2d", self.nn_name2id)
        layer_outputs = [pool2d_name, output_tensor_name]

        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool2d_options = Pool2DOptions()
        pool2d_options.Init(op_options.Bytes, op_options.Pos)
        stride_h = pool2d_options.StrideH()
        stride_w = pool2d_options.StrideW()
        padding = pool2d_options.Padding()
        filter_h = pool2d_options.FilterHeight()
        filter_w = pool2d_options.FilterWidth()
        fused_activation_fn = pool2d_options.FusedActivationFunction()

        layer_attrs = {
            "kernel_size": (filter_h, filter_w),
            "stride": (stride_h, stride_w),
            "padding": [0, 0],
        }

        if padding == Padding.VALID:
            layer_attrs["padding"] = string("VALID")
        elif padding == Padding.SAME:
            layer_attrs["padding"] = string("SAME")
            
        if input_tensor.qnn_params:
            dequant_name = name_generator("dequant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:DequantizeLinear",
                inputs={"x": input_tensor_name},
                outputs=[dequant_name, input_tensor_name + "_quant"],
                zero_point=input_tensor.qnn_params["zero_point"],
                scale=input_tensor.qnn_params["scale"])
            input_tensor_name = input_tensor_name + "_quant"

        self.paddle_graph.add_layer(
            kernel="paddle.transpose",
            inputs={"x": input_tensor_name},
            outputs=[input_tensor_name + "_transpose"],
            perm=[0, 3, 1, 2])
        self.paddle_graph.add_layer(
            kernel="paddle.nn.MaxPool2D"
            if pool_type == "max" else "paddle.nn.AvgPool2D",
            inputs={"input": input_tensor_name + "_transpose"},
            outputs=layer_outputs,
            **layer_attrs)
        self.paddle_graph.add_layer(
            kernel="paddle.transpose",
            inputs={"x": output_tensor_name},
            outputs=[output_tensor_name],
            perm=[0, 2, 3, 1])

        self.convert_fused_activation_function(fused_activation_fn,
                                               output_tensor_name)
        
        if output_tensor.qnn_params:
            dtype = output_tensor.tensor.Type()
            quant_name = name_generator("quant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:QuantizeLinear",
                inputs={"x": output_tensor_name},
                outputs=[quant_name, output_tensor_name],
                zero_point=output_tensor.qnn_params["zero_point"],
                scale=output_tensor.qnn_params["scale"],
                dtype=dtype)

    def convert_conv2d(self, op, conv_type):
        from tflite.Conv2DOptions import Conv2DOptions
        from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
        from tflite.Padding import Padding

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"
        input_tensor = input_tensors[0]
        _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
        input_tensor_name = input_tensor.tensor.Name().decode("utf8")
        input_tensor_name = rename(input_tensor_name)
        weight_tensor = input_tensors[1]
        output_channels, kernel_h, kernel_w, input_channels = weight_tensor.tensor.ShapeAsNumpy(
        )

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_name = output_tensor.tensor.Name().decode("utf8")
        output_tensor_name = rename(output_tensor_name)

        conv2d_name = name_generator("cov2d", self.nn_name2id)
        layer_outputs = [conv2d_name, output_tensor_name]
        self.params[conv2d_name + ".weight"] = np.transpose(
            self.get_tensor_value(weight_tensor), (0, 3, 1, 2))
        
        if weight_tensor.qnn_params:
            kernel_zero_point = weight_tensor.qnn_params["zero_point"]
            kernel_scale = weight_tensor.qnn_params["scale"]
            self.params[conv2d_name + ".weight"] = self.params[conv2d_name + ".weight"].astype("float32")
            self.params[conv2d_name + ".weight"] = (self.params[conv2d_name + ".weight"] - kernel_zero_point) * kernel_scale
        
        if conv_type == "conv2d":
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == "dw_conv2d":
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
            depth_multiplier = conv_options.DepthMultiplier()
            real_input_channels = input_channels * depth_multiplier

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        layer_attrs = {
            "in_channels": input_channels,
            "out_channels": output_channels,
            "kernel_size": [kernel_h, kernel_w]
            if kernel_h != kernel_w else kernel_h,
            "stride": [stride_h, stride_w]
            if stride_h != stride_w else stride_h,
            "padding": [0, 0],
            "dilation": [dilation_h, dilation_w]
            if dilation_h != dilation_w else dilation_h,
        }
        
        if conv_type == "dw_conv2d":
            self.params[conv2d_name + ".weight"] = np.transpose(
                self.params[conv2d_name + ".weight"], (1, 0, 2, 3))
            layer_attrs["groups"] = input_channels
            layer_attrs["in_channels"] = real_input_channels
            layer_attrs["out_channels"] = output_channels * layer_attrs["groups"]

        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            self.params[conv2d_name + ".bias"] = self.get_tensor_value(
                bias_tensor)
            if bias_tensor.qnn_params:
                bias_zero_point = bias_tensor.qnn_params["zero_point"]
                bias_scale = bias_tensor.qnn_params["scale"]
                self.params[conv2d_name + ".bias"] = self.params[conv2d_name + ".bias"].astype("float32")
                self.params[conv2d_name + ".bias"] = (self.params[conv2d_name + ".bias"] - bias_zero_point) * bias_scale
        else:
            layer_attrs["bias_attr"] = False

        if padding == Padding.VALID:
            layer_attrs["padding"] = string("VALID")
        elif padding == Padding.SAME:
            layer_attrs["padding"] = string("SAME")
            
        if input_tensor.qnn_params:
            dequant_name = name_generator("dequant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:DequantizeLinear",
                inputs={"x": input_tensor_name},
                outputs=[dequant_name, input_tensor_name + "_quant"],
                zero_point=input_tensor.qnn_params["zero_point"],
                scale=input_tensor.qnn_params["scale"])
            input_tensor_name = input_tensor_name + "_quant"

        self.paddle_graph.add_layer(
            kernel="paddle.transpose",
            inputs={"x": input_tensor_name},
            outputs=[input_tensor_name + "_transpose"],
            perm=[0, 3, 1, 2])
        self.paddle_graph.add_layer(
            kernel="paddle.nn.Conv2D",
            inputs={"input": input_tensor_name + "_transpose"},
            outputs=layer_outputs,
            **layer_attrs)
        self.paddle_graph.add_layer(
            kernel="paddle.transpose",
            inputs={"x": output_tensor_name},
            outputs=[output_tensor_name],
            perm=[0, 2, 3, 1])

        self.convert_fused_activation_function(fused_activation_fn,
                                               output_tensor_name)
        
        if output_tensor.qnn_params:
            dtype = output_tensor.tensor.Type()
            quant_name = name_generator("quant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:QuantizeLinear",
                inputs={"x": output_tensor_name},
                outputs=[quant_name, output_tensor_name],
                zero_point=output_tensor.qnn_params["zero_point"],
                scale=output_tensor.qnn_params["scale"],
                dtype=dtype)

    def AVERAGE_POOL_2D(self, op):
        self.convert_pool2d(op, pool_type="average")

    def MAX_POOL_2D(self, op):
        self.convert_pool2d(op, pool_type="max")

    def CONCATENATION(self, op):
        from tflite.ConcatenationOptions import ConcatenationOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors should greater than 1"
        input_tensors_name = list()
        for i in range(len(input_tensors)):
            input_tensors_name.append(
                rename(input_tensors[i].tensor.Name().decode("utf8")))

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_name = output_tensor.tensor.Name().decode("utf8")
        output_tensor_name = rename(output_tensor_name)

        assert op.BuiltinOptionsType() == BuiltinOptions.ConcatenationOptions
        op_options = op.BuiltinOptions()
        concatenation_options = ConcatenationOptions()
        concatenation_options.Init(op_options.Bytes, op_options.Pos)
        concatenation_axis = concatenation_options.Axis()
        fused_activation_fn = concatenation_options.FusedActivationFunction()

        self.paddle_graph.add_layer(
            kernel="paddle.concat",
            inputs={"x": input_tensors_name},
            outputs=[output_tensor_name],
            axis=concatenation_axis)

        self.convert_fused_activation_function(fused_activation_fn,
                                               output_tensor_name)

    def RESHAPE(self, op):
        from tflite.ReshapeOptions import ReshapeOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (1, 2), "input tensors should not be empty"
        input_tensor = input_tensors[0]
        input_tensor_name = input_tensor.tensor.Name().decode("utf8")
        input_tensor_name = rename(input_tensor_name)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "There should be only 1 output tensor"
        output_tensor = output_tensors[0]
        output_tensor_name = output_tensors[0].tensor.Name().decode("utf8")
        output_tensor_name = rename(output_tensor_name)

        if len(input_tensors) == 2:
            shape_tensor = input_tensors[1]
            target_shape = self.get_tensor_value(shape_tensor)
            l = list()
            for i in target_shape:
                l.append(int(i))
            target_shape = l
        else:
            assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
            op_options = op.BuiltinOptions()
            reshape_options = ReshapeOptions()
            reshape_options.Init(op_options.Bytes, op_options.Pos)
            target_shape = reshape_options.NewShapeAsNumpy()
            target_shape = [int(x) for x in target_shape]
            
        if input_tensor.qnn_params:
            dequant_name = name_generator("dequant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:DequantizeLinear",
                inputs={"x": input_tensor_name},
                outputs=[dequant_name, input_tensor_name + "_quant"],
                zero_point=input_tensor.qnn_params["zero_point"],
                scale=input_tensor.qnn_params["scale"])
            input_tensor_name = input_tensor_name + "_quant"

        self.paddle_graph.add_layer(
            kernel="paddle.reshape",
            inputs={"x": input_tensor_name},
            outputs=[output_tensor_name],
            shape=target_shape)
        
        if output_tensor.qnn_params:
            dtype = output_tensor.tensor.Type()
            quant_name = name_generator("quant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:QuantizeLinear",
                inputs={"x": output_tensor_name},
                outputs=[quant_name, output_tensor_name],
                zero_point=output_tensor.qnn_params["zero_point"],
                scale=output_tensor.qnn_params["scale"],
                dtype=dtype)

    def SOFTMAX(self, op):
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_name = input_tensor.tensor.Name().decode("utf8")
        input_tensor_name = rename(input_tensor_name)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "There should be only 1 output tensor"
        output_tensor = output_tensors[0]
        output_tensor_name = output_tensor.tensor.Name().decode("utf8")
        output_tensor_name = rename(output_tensor_name)

        softmax_name = name_generator("softmax", self.nn_name2id)
        layer_outputs = [softmax_name, output_tensor_name]
        
        if input_tensor.qnn_params:
            dequant_name = name_generator("dequant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:DequantizeLinear",
                inputs={"x": input_tensor_name},
                outputs=[dequant_name, input_tensor_name + "_quant"],
                zero_point=input_tensor.qnn_params["zero_point"],
                scale=input_tensor.qnn_params["scale"])
            input_tensor_name = input_tensor_name + "_quant"

        self.paddle_graph.add_layer(
            kernel="paddle.nn.Softmax",
            inputs={"x": input_tensor_name},
            outputs=layer_outputs)
        
        if output_tensor.qnn_params:
            dtype = output_tensor.tensor.Type()
            quant_name = name_generator("quant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:QuantizeLinear",
                inputs={"x": output_tensor_name},
                outputs=[quant_name, output_tensor_name],
                zero_point=output_tensor.qnn_params["zero_point"],
                scale=output_tensor.qnn_params["scale"],
                dtype=dtype)
        
        
    def DEPTHWISE_CONV_2D(self, op):
        self.convert_conv2d(op, "dw_conv2d")

    
    def CONV_2D(self, op):
        self.convert_conv2d(op, "conv2d")
        
        
    def ADD(self, op):
        from tflite.AddOptions import AddOptions
        
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        lhs_tensor = input_tensors[0]
        rhs_tensor = input_tensors[1]
        lhs_tensor_name = lhs_tensor.tensor.Name().decode("utf8")
        lhs_tensor_name = rename(lhs_tensor_name)
        rhs_tensor_name = rhs_tensor.tensor.Name().decode("utf8")
        rhs_tensor_name = rename(rhs_tensor_name)
        
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_name = output_tensor.tensor.Name().decode("utf8")
        output_tensor_name = rename(output_tensor_name)
        
        assert op.BuiltinOptionsType() == BuiltinOptions.AddOptions
        op_options = op.BuiltinOptions()
        add_options = AddOptions()
        add_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = add_options.FusedActivationFunction()
        
        if lhs_tensor.qnn_params:
            
            dequant_name = name_generator("dequant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:DequantizeLinear",
                inputs={"x": lhs_tensor_name},
                outputs=[dequant_name, lhs_tensor_name + "_quant"],
                zero_point=lhs_tensor.qnn_params["zero_point"],
                scale=lhs_tensor.qnn_params["scale"])
            lhs_tensor_name = lhs_tensor_name + "_quant"
            dequant_name = name_generator("dequant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:DequantizeLinear",
                inputs={"x": rhs_tensor_name},
                outputs=[dequant_name, rhs_tensor_name + "_quant"],
                zero_point=rhs_tensor.qnn_params["zero_point"],
                scale=rhs_tensor.qnn_params["scale"])
            rhs_tensor_name = rhs_tensor_name + "_quant"
            
        self.paddle_graph.add_layer(
            kernel="paddle.add",
            inputs={"x": lhs_tensor_name,
                    "y": rhs_tensor_name},
            outputs=[output_tensor_name])
        
        self.convert_fused_activation_function(fused_activation_fn,
                                               output_tensor_name)
        
        if output_tensor.qnn_params:
            dtype = output_tensor.tensor.Type()
            quant_name = name_generator("quant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:QuantizeLinear",
                inputs={"x": output_tensor_name},
                outputs=[quant_name, output_tensor_name],
                zero_point=output_tensor.qnn_params["zero_point"],
                scale=output_tensor.qnn_params["scale"],
                dtype=dtype)

    def FULLY_CONNECTED(self, op):
        from tflite.FullyConnectedOptions import FullyConnectedOptions
        
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (2, 3), "input tensors length should be two or three"
        input_tensor = input_tensors[0]
        input_n, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
        input_tensor_name = input_tensor.tensor.Name().decode("utf8")
        input_tensor_name = rename(input_tensor_name)
        weight_tensor = input_tensors[1]
        output_channels, input_channels = weight_tensor.tensor.ShapeAsNumpy()
        
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_name = output_tensor.tensor.Name().decode("utf8")
        output_tensor_name = rename(output_tensor_name)

        fc_name = name_generator("fc", self.nn_name2id)
        layer_outputs = [fc_name, output_tensor_name]
        self.params[fc_name + ".weight"] = np.transpose(self.get_tensor_value(weight_tensor), (1, 0))
            
        if weight_tensor.qnn_params:
            kernel_zero_point = weight_tensor.qnn_params["zero_point"]
            kernel_scale = weight_tensor.qnn_params["scale"]
            self.params[fc_name + ".weight"] = self.params[fc_name + ".weight"].astype("float32")
            self.params[fc_name + ".weight"] = (self.params[fc_name + ".weight"] - kernel_zero_point) * kernel_scale
            
        layer_attrs = dict()
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            self.params[fc_name + ".bias"] = self.get_tensor_value(
                bias_tensor)
            if bias_tensor.qnn_params:
                bias_zero_point = bias_tensor.qnn_params["zero_point"]
                bias_scale = bias_tensor.qnn_params["scale"]
                self.params[fc_name + ".bias"] = self.params[fc_name + ".bias"].astype("float32")
                self.params[fc_name + ".bias"] = (self.params[fc_name + ".bias"] - bias_zero_point) * bias_scale
        else:
            layer_attrs["bias_attr"] = False
            
        assert op.BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions
        op_options = op.BuiltinOptions()
        fully_connected_options = FullyConnectedOptions()
        fully_connected_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = fully_connected_options.FusedActivationFunction()
        layer_attrs["in_features"] = input_channels
        layer_attrs["out_features"] = output_channels
        
        if input_tensor.qnn_params:
            dequant_name = name_generator("dequant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:DequantizeLinear",
                inputs={"x": input_tensor_name},
                outputs=[dequant_name, input_tensor_name + "_quant"],
                zero_point=input_tensor.qnn_params["zero_point"],
                scale=input_tensor.qnn_params["scale"])
            input_tensor_name = input_tensor_name + "_quant"

        self.paddle_graph.add_layer(
            kernel="paddle.reshape",
            inputs={"x": input_tensor_name},
            outputs=[input_tensor_name + "_reshape"],
            shape=[input_n, input_c])
        self.paddle_graph.add_layer(
            kernel="paddle.nn.Linear",
            inputs={"input": input_tensor_name + "_reshape"},
            outputs=layer_outputs,
            **layer_attrs)

        self.convert_fused_activation_function(fused_activation_fn,
                                               output_tensor_name)
        
        if output_tensor.qnn_params:
            dtype = output_tensor.tensor.Type()
            quant_name = name_generator("quant", self.nn_name2id)
            self.paddle_graph.add_layer(
                kernel="custom_layer:QuantizeLinear",
                inputs={"x": output_tensor_name},
                outputs=[quant_name, output_tensor_name],
                zero_point=output_tensor.qnn_params["zero_point"],
                scale=output_tensor.qnn_params["scale"],
                dtype=dtype)