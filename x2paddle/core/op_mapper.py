#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.proto import framework_pb2
from x2paddle.core.util import *
import inspect
import os
import os.path as osp
import pickle


def export_paddle_param(param, param_name, dir):
    dtype_map = {
        "int16": [framework_pb2.VarType.INT16, 'h'],
        "int32": [framework_pb2.VarType.INT32, 'i'],
        "int64": [framework_pb2.VarType.INT64, 'q'],
        "float16": [framework_pb2.VarType.FP16, 'e'],
        "float32": [framework_pb2.VarType.FP32, 'f'],
        "float64": [framework_pb2.VarType.FP64, 'd'],
        "bool": [framework_pb2.VarType.BOOL, None]
    }
    shape = param.shape
    if len(shape) == 0:
        assert param.size == 1, "Unexpected situation happend!"
        shape = [1]
    assert str(param.dtype) in dtype_map, "Unknown dtype of params."

    fp = open(os.path.join(dir, param_name), 'wb')
    numpy.array([0], dtype='int32').tofile(fp)
    numpy.array([0], dtype='int64').tofile(fp)
    numpy.array([0], dtype='int32').tofile(fp)
    tensor_desc = framework_pb2.VarType.TensorDesc()
    tensor_desc.data_type = dtype_map[str(param.dtype)][0]
    tensor_desc.dims.extend(shape)
    desc_size = tensor_desc.ByteSize()
    numpy.array([desc_size], dtype='int32').tofile(fp)
    fp.write(tensor_desc.SerializeToString())
    param.tofile(fp)
    fp.close()


# This func will copy to generate code file
def run_net(param_dir="./"):
    import os
    inputs, outputs = x2paddle_net()
    for i, out in enumerate(outputs):
        if isinstance(out, list):
            for out_part in out:
                outputs.append(out_part)
            del outputs[i]
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        b = os.path.exists(os.path.join(param_dir, var.name))
        return b

    fluid.io.load_vars(
        exe, param_dir, fluid.default_main_program(), predicate=if_exist)


def run_net_dygraph(param_dir="./", *args):
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        restore, _ = fluid.load_dygraph(oa.path.join(param_dir, 'model'))
        model = X2PaddleNet()
        model.set_dict(restore)
        out = model(*args)


class OpMapper(object):
    def __init__(self):
        self.paddle_codes = ""
        self.tab = "    "
        self.net_code = list()
        self.weights = dict()
        self.inputs = list()
        self.outputs = list()
        from x2paddle.op_mapper.pytorch_op_mapper import PyTorchOpMapper
        self.is_dygraph = True if isinstance(self, PyTorchOpMapper) else False
        if self.is_dygraph:
            self.paddle_codes_init = ""
            self.paddle_codes_forward = ""

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self, op):
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            print("There are {} ops not supported yet, list as below".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print(op)
            return False

    def add_codes(self, codes, indent=0):
        if isinstance(codes, list):
            for code in codes:
                self.paddle_codes += (
                    self.tab * indent + code.strip('\n') + '\n')
        elif isinstance(codes, str):
            self.paddle_codes += (self.tab * indent + codes.strip('\n') + '\n')
        else:
            raise Exception("Unknown type of codes")

    def add_codes_dygraph(self, codes, init_indent=0, forward_indent=0):
        if isinstance(codes, list):
            for code in codes:
                if code.startswith(
                        'self.') or 'init' in code or 'super' in code:
                    self.paddle_codes_init += (
                        self.tab * init_indent + code.strip('\n') + '\n')
                else:
                    self.paddle_codes_forward += (
                        self.tab * forward_indent + code.strip('\n') + '\n')
        elif isinstance(codes, str):
            if 'init' in codes or 'super' in codes:
                self.paddle_codes_init += (
                    self.tab * init_indent + codes.strip('\n') + '\n')
            else:
                self.paddle_codes_forward += (
                    self.tab * forward_indent + codes.strip('\n') + '\n')
        else:
            raise Exception("Unknown type of codes")

    def add_heads(self):
        self.add_codes("from paddle.fluid.initializer import Constant")
        self.add_codes("from paddle.fluid.param_attr import ParamAttr")
        self.add_codes("import paddle.fluid as fluid")
        if self.is_dygraph:
            self.add_codes("import math")
            self.add_codes("import functools")
        self.add_codes("")

    def save_inference_model(self, save_dir, params_merge):
        self.save_python_model(save_dir)

        import sys
        import paddle.fluid as fluid
        py_code_dir = os.path.join(save_dir, "model_with_code")
        sys.path.append(py_code_dir)
        import model
        try:
            inputs, outputs = model.x2paddle_net()
            for i, out in enumerate(outputs):
                if isinstance(out, list):
                    for out_part in out:
                        outputs.append(out_part)
                    del outputs[i]
            input_names = [input.name for input in inputs]
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            def if_exist(var):
                b = os.path.exists(
                    os.path.join(os.path.join(py_code_dir, var.name)))
                return b

            fluid.io.load_vars(
                exe,
                py_code_dir,
                fluid.default_main_program(),
                predicate=if_exist)
            if params_merge:
                fluid.io.save_inference_model(
                    dirname=os.path.join(save_dir, "inference_model"),
                    feeded_var_names=input_names,
                    target_vars=outputs,
                    executor=exe,
                    params_filename="__params__")
            else:
                fluid.io.save_inference_model(
                    dirname=os.path.join(save_dir, "inference_model"),
                    feeded_var_names=input_names,
                    target_vars=outputs,
                    executor=exe,
                    params_filename=None)
        except:
            raise Exception(
                "Paddle code was saved in {}/model.py, but seems there's wrong exist, please check model.py manually."
                .format(py_code_dir))

    def save_python_model(self, save_dir):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        py_code_dir = os.path.join(save_dir, "model_with_code")
        if not os.path.exists(py_code_dir):
            os.makedirs(py_code_dir)

        if self.is_dygraph:
            params_output = open(
                os.path.join(py_code_dir, 'model.pdparams'), 'wb')
            pickle.dump(self.weights, params_output)
        else:
            for name, param in self.weights.items():
                export_paddle_param(param, name, py_code_dir)
        self.add_heads()

        if hasattr(self, "used_custom_layers"):
            for _, layer_code in self.used_custom_layers.items():
                self.add_codes(layer_code, 0)
                self.add_codes("", 0)

        if self.is_dygraph:
            self.add_codes("class X2PaddleNet(fluid.dygraph.Layer):", 0)
            self.add_codes_dygraph("\ndef __init__(self):", 1)
            self.add_codes_dygraph("\nsuper(X2PaddleNet, self).__init__()", 2)
            self.add_codes_dygraph(
                "\ndef forward(self, {}):".format(','.join(self.datas_name)), forward_indent=1)
            net_indent = 2
            self.node_net_indent = {}
        else:
            self.add_codes("\ndef x2paddle_net():", 0)
            net_indent = 1
        
        for i in range(len(self.graph.topo_sort)):
            node_name = self.graph.topo_sort[i]
            node = self.graph.get_node(node_name)
            if node is None:
                continue
            if len(node.fluid_code.layers) == 0:
                continue
            if self.is_dygraph:
                from x2paddle.decoder.pytorch_decoder import PyTorchGraphControlNode
                if isinstance(node, PyTorchGraphControlNode):
                    block_indent = net_indent + 1
                    if node.layer_name in self.node_net_indent:
                        block_indent = self.node_net_indent[node.layer_name] + 1
                    for node_name in node.block_nodes_name:
                        node_name = node_name.replace('/', '_').replace('-', '_').replace('.', '_').replace('%', 'x_')
                        self.node_net_indent[node_name] = block_indent
                net_init_indent = net_indent
                net_forward_indent = net_indent
                if node.layer_name in self.node_net_indent:
                    net_forward_indent = self.node_net_indent[node.layer_name]
                self.add_codes_dygraph(node.fluid_code.gen_codes(), net_init_indent, net_forward_indent)
            else:
                self.add_codes(node.fluid_code.gen_codes(), net_indent)
        self.add_codes(self.paddle_codes_init)
        self.add_codes("", 0)
        self.add_codes(self.paddle_codes_forward)

        input_str = "["
        for name in self.graph.input_nodes:
            input_str += (name + ", ")
        input_str = input_str.strip(", ") + "]"
        output_str = "["
        for name in self.graph.output_nodes:
            output_str += (name + ", ")
        output_str = output_str.strip(", ") + "]"

        return_code = "return {}, {}".format(
            input_str.replace('/', '_').replace('-', '_').replace(
                '.', '_').replace('%', 'x_'),
            output_str.replace('/', '_').replace('-', '_').replace(
                '.', '_').replace('%', 'x_'))

        self.add_codes(return_code, net_indent)
        self.add_codes("", 0)
        if self.is_dygraph:
            self.add_codes(inspect.getsourcelines(run_net_dygraph)[0])
        else:
            self.add_codes(inspect.getsourcelines(run_net)[0])
        fp = open(os.path.join(py_code_dir, "model.py"), 'w')
        fp.write(self.paddle_codes)
        fp.close()
