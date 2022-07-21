# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import logging
import paddle
import onnx
from onnx import helper
from onnx import TensorProto
from onnxruntime import InferenceSession

DTYPE_ONNX_STR_MAP = {
    'float32': TensorProto.FLOAT,
    'float64': TensorProto.DOUBLE,
    'int16': TensorProto.INT16,
    'int32': TensorProto.INT32,
    'int64': TensorProto.INT64,
    'bool': TensorProto.BOOL,
}


def compare(result, expect, delta=1e-10, rtol=1e-10):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if type(result) == np.ndarray:
        if type(expect) == list:
            expect = expect[0]
        expect = np.array(expect)
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            if result.dtype == np.bool_:
                diff = abs(result.astype("int32") - expect.astype("int32"))
            else:
                diff = abs(result - expect)
            logging.error("Output has diff! max diff: {}".format(np.amax(diff)))
        if result.dtype != expect.dtype:
            logging.error(
                "Different output data types! res type is: {}, and expect type is: {}".
                format(result.dtype, expect.dtype))
        assert res
        assert result.shape == expect.shape, "result.shape: {} != expect.shape: {}".format(
            result.shape, expect.shape)
        assert result.dtype == expect.dtype, "result.dtype: {} != expect.dtype: {}".format(
            result.dtype, expect.dtype)
    elif isinstance(result, (list, tuple)) and len(result) > 1:
        for i in range(len(result)):
            if isinstance(result[i], (np.generic, np.ndarray)):
                compare(result[i], expect[i], delta, rtol)
            else:
                compare(result[i].numpy(), expect[i], delta, rtol)
    elif len(result) == 1:
        compare(result[0], expect[0], delta, rtol)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)

    elif dtype == "bool":
        return np.random.randint(low, high, shape).astype("bool")


class ONNXConverter(object):
    """
     onnx model transfer to paddle
    """

    def __init__(self,
                 file_name,
                 ver_list,
                 op_type=[],
                 inputs_name=[],
                 outputs_name=[],
                 inputs_shape=[],
                 outputs_shape=[],
                 outputs_dtype=[],
                 delta=1e-5,
                 rtol=1e-5,
                 use_gpu=True,
                 attrs=[]):
        self.op_type = op_type
        assert isinstance(self.op_type,
                          str), "The dtype of op_type must be string!"
        self.seed = 33
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        if use_gpu and paddle.device.is_compiled_with_cuda() is True:
            self.places = ['gpu']
        else:
            self.places = ['cpu']
        self.name = file_name
        self._version = ver_list
        self.pwd = os.getcwd()
        self.delta = delta
        self.rtol = rtol
        self.static = False
        self.kwargs_dict = {"input_data": ()}
        self.input_feed = {}
        self.inputs_dtype = []
        self.inputs_name = inputs_name
        self.outputs_name = outputs_name
        self.inputs_shape = inputs_shape
        self.outputs_shape = outputs_shape
        self.outputs_dtype = outputs_dtype
        self.attrs = attrs

    def set_input_data(self, group_name, *args):
        """
        set input data
        """
        self.kwargs_dict[group_name] = args
        if isinstance(self.kwargs_dict[group_name][0], tuple):
            self.kwargs_dict[group_name] = self.kwargs_dict[group_name][0]

        i = 0
        for in_data in self.kwargs_dict[group_name]:
            if isinstance(in_data, list):
                for data in in_data:
                    self.inputs_dtype.append(str(data.dtype))
                    self.input_feed[self.inputs_name[i]] = data
                    i += 1
            else:
                if isinstance(in_data, tuple):
                    in_data = in_data[0]
                self.inputs_dtype.append(str(in_data.dtype))
                self.input_feed[self.inputs_name[i]] = in_data
                i += 1

    def _mkdir(self):
        """
        make dir to save all
        """
        save_path = os.path.join(self.pwd, self.name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def _onnx_to_paddle(self, ver):
        """
        convert onnx to paddle
        """
        from x2paddle.convert import onnx2paddle
        onnx_path = os.path.join(self.pwd, self.name,
                                 self.name + '_' + str(ver) + '.onnx')
        paddle_path = os.path.join(self.pwd, self.name,
                                   self.name + '_' + str(ver) + '_paddle')
        onnx2paddle(onnx_path, paddle_path, convert_to_lite=False)

    def _mk_paddle_res(self, ver):
        """
        make paddle res
        """
        paddle_path = os.path.join(
            self.pwd, self.name,
            self.name + '_' + str(ver) + '_paddle/inference_model/model')
        paddle.disable_static()
        # run
        model = paddle.jit.load(paddle_path)
        paddle_feed = list()

        for i in range(len(self.input_feed)):
            paddle_feed.append(self.input_feed[self.inputs_name[i]])
        result = model(*paddle_feed)
        # get paddle outputs
        if isinstance(result, (tuple, list)):
            result = tuple(out.numpy() for out in result)
        else:
            result = (result.numpy(), )
        print("paddle result:", result[0].shape)
        return result

    def _mk_onnx_res(self, ver):
        """
        make onnx res
        """
        sess = InferenceSession(
            os.path.join(self.pwd, self.name, self.name + '_' + str(ver) +
                         '.onnx'))
        ort_outs = sess.run(output_names=None, input_feed=self.input_feed)
        print("onnx result:", ort_outs[0].shape)
        return ort_outs

    def set_onnx_inputs(self):
        graph_inputs = list()
        for i in range(len(self.inputs_name)):
            graph_inputs.append(
                helper.make_tensor_value_info(self.inputs_name[
                    i], DTYPE_ONNX_STR_MAP[self.inputs_dtype[i]],
                                              self.inputs_shape[i]))

        return graph_inputs

    def set_onnx_outputs(self):
        graph_outputs = list()
        for i in range(len(self.outputs_name)):
            graph_outputs.append(
                helper.make_tensor_value_info(self.outputs_name[
                    i], DTYPE_ONNX_STR_MAP[self.outputs_dtype[i][0]],
                                              self.outputs_shape[i]))

        return graph_outputs

    def _mk_onnx_graph(self, ver):
        """
        make onnx graph
        """
        node = onnx.helper.make_node(
            self.op_type,
            inputs=self.inputs_name,
            outputs=self.outputs_name,
            **self.attrs, )
        graph_inputs = self.set_onnx_inputs()
        graph_outputs = self.set_onnx_outputs()
        graph = helper.make_graph(
            [node],
            self.name,
            graph_inputs,  # graph inputs
            graph_outputs,  # graph outputs
        )
        opset_imports = [helper.make_opsetid("", ver)]
        model = helper.make_model(
            graph, producer_name='onnx-example', opset_imports=opset_imports)
        onnx.save(model,
                  os.path.join(self.pwd, self.name,
                               self.name + '_' + str(ver) + '.onnx'))
        onnx.checker.check_model(model)

    def run(self):
        """
        1. make onnx model
        2. convert onnx to paddle
        3. use onnx to make res
        4. compare diff
        """
        self._mkdir()
        for place in self.places:
            paddle.set_device(place)
            onnx_res = {}
            paddle_res = {}
            # export onnx models and make onnx res
            for v in self._version:
                self._mk_onnx_graph(ver=v)
                self._onnx_to_paddle(ver=v)
                onnx_res[str(v)] = self._mk_onnx_res(ver=v)
                paddle_res[str(v)] = self._mk_paddle_res(ver=v)

            for v in self._version:
                compare(
                    onnx_res[str(v)],
                    paddle_res[str(v)],
                    delta=self.delta,
                    rtol=self.rtol)