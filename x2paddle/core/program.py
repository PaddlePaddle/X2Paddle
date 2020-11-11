# -*- coding:UTF-8 -*-
#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
from __future__ import division
import paddle.fluid as fluid
import os.path as osp
import paddle
from paddle.fluid.proto import framework_pb2
from collections import OrderedDict
import numpy
import collections
import sys
import os
import six
import pickle


class PaddleLayer(object):
    def __init__(self, id, kernel, inputs, outputs, **kwargs):
        assert isinstance(
            inputs,
            dict), "parameter 'inputs' for PaddleLayer should be type of dict"
        assert isinstance(
            outputs,
            list), "parameter 'outputs' for PaddleLayer should be type of list"
        for k, v in inputs.items():
            if isinstance(v, list):
                for i in v:
                    assert isinstance(
                        i, six.string_types
                    ), "value in inputs should be type of string or list of string"
            else:
                assert isinstance(v, six.string_types) or isinstance(
                    v, list
                ), "value in inputs should be type of string or list of string"
        for v in outputs:
            assert isinstance(
                v, six.
                string_types), "elements in outputs should be type of string"
        self.kernel = kernel
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = kwargs
        self.id = id
        self.blocks = list()

    def add_block(self, block):
        self.blocks.append(block)


class PaddleGraph(object):
    def __init__(self, parent_layer=None, graph_type="static"):
        self.layers = OrderedDict()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()
        self.parent_layer = parent_layer
        self.graph_type = graph_type

    def set_name(self, name):
        self.name = name

    def set_parameters(self, parameters):
        self.parameters = parameters

    def clear(self):
        self.layers = OrderedDict()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()

    def clear_edges(self):
        self.edges_out = dict()
        self.edges_in = dict()

    def add_layer(self, kernel, inputs, outputs, **kwargs):
        layer_id = str(len(self.layers))
        if self.parent_layer is not None:
            layer_id = "{}.{}.{}".format(self.parent_layer.id,
                                         len(self.parent_layer.blocks),
                                         layer_id)
        layer = PaddleLayer(layer_id, kernel, inputs, outputs, **kwargs)
        self.layers[layer_id] = layer
        return layer_id

    def del_layer(self, layer_id):
        layer = self.layers[layer_id]
        outputs = self.edges_out.get(layer_id, [])
        inputs = self.edges_in.get(layer_id, [])

        assert len(
            inputs) <= 1, "There should be 0 or 1 input for deleted layer."

        if len(inputs) == 0:
            for out in outputs:
                while layer_id in self.edges_in[out]:
                    index = self.edges_in[out].index(layer_id)
                    del self.edges_in[out][index]

                input_keys = list(self.layers[out].inputs.keys())
                for k in input_keys:
                    if self.layers[out].inputs[k] == layer.outputs[0]:
                        del self.layers[out].inputs[k]

            del self.layers[layer_id]
            if layer_id in self.edges_in:
                del self.edges_in[layer_id]
            if layer_id in self.edges_out:
                del self.edges_out[layer_id]
            return

        # 将所有输出layer的输入layer进行替换
        for out in outputs:
            for i in range(len(self.edges_in[out])):
                if self.edges_in[out][i] == layer_id:
                    self.edges_in[out][i] = inputs[0]

        # 将输出layer赋给输入layer的输出
        replace_index = self.edges_out[inputs[0]].index(layer_id)
        del self.edges_out[inputs[0]][replace_index]
        for i, out in enumerate(outputs):
            self.edges_out[inputs[0]].insert(replace_index + i, out)
            for k, v in self.layers[out].inputs.items():
                if v == layer.outputs[0]:
                    self.layers[out].inputs[k] = list(layer.inputs.values())[0]

        del self.layers[layer_id]
        if layer_id in self.edges_out:
            del self.edges_out[layer_id]
        if layer_id in self.edges_in:
            del self.edges_in[layer_id]

    def build(self, inputs=None, outputs=None):
        self.clear_edges()
        outputs_from_nodes = dict()
        for layer_id, layer in self.layers.items():
            for input_key, input_var in layer.inputs.items():
                vs = input_var
                if not isinstance(vs, list):
                    vs = [vs]
                for v in vs:
                    assert v in outputs_from_nodes or (
                        inputs is not None and v in list(inputs.values())
                    ) or (
                        outputs is not None and v in outputs
                    ), "Couldn't find {} in previous layers, the layers should be make by topological sort".format(
                        v)
                    if v in outputs_from_nodes:
                        in_layer_id = outputs_from_nodes[v]
                    else:
                        in_layer_id = -1
                    if in_layer_id not in self.edges_out:
                        self.edges_out[in_layer_id] = list()
                    self.edges_out[in_layer_id].append(layer_id)

                    if layer_id not in self.edges_in:
                        self.edges_in[layer_id] = list()
                    self.edges_in[layer_id].append(in_layer_id)
            for output in layer.outputs:
                outputs_from_nodes[output] = layer_id

            # 将block的输出用于父图
            if inputs is not None and outputs is not None and set(
                    layer.outputs).issubset(outputs):
                if layer_id not in self.edges_out:
                    self.edges_out[layer_id] = list()
                self.edges_out[layer_id].append(-1)

            # 处理子图
            if len(layer.blocks) > 0:
                for block in layer.blocks:
                    block.build(layer.inputs, layer.outputs)

        # 删除不必要的节点
        invalid_list = list()
        for layer_id, layer in self.layers.items():
            if len(self.layers) > 1:
                if self.edges_in.get(layer_id, 0) == 0 and self.edges_out.get(
                        layer_id, 0) == 0 and layer.kernel != "prim.assert" \
                        and layer.kernel != "prim.exception" \
                        and layer.kernel != "prim.warnings":
                    invalid_list.append(layer_id)
        for layer_id in invalid_list:
            self.layers.pop(layer_id)

        if self.graph_type == "dygraph":
            self.get_dygraph_inputs()
            if len(self.outputs) == 0:
                self.get_dygraph_outputs()

    def get_global_layers(self):
        # 该全局layers的信息是按照拓扑排序组成的
        def update(layers):
            global_layers = dict()
            for layer_id, layer in layers.items():
                global_layers[layer_id] = layer
                for block in layer.blocks:
                    block_global_layers = update(block.layers)
                    global_layers.update(block_global_layers)
            return global_layers

        return update(self.layers)

    def gen_code(self, code_dir):
        def write_code(f, code_list, indent=0):
            indent_blank = "    " * indent
            for code_line in code_list:
                if code_line.strip() == "":
                    f.write('\n')
                else:
                    f.write(indent_blank + code_line + '\n')

        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        f = open(os.path.join(code_dir, 'x2paddle_model.py'), 'w')

        write_code(
            f, [
                "from paddle.fluid.initializer import Constant",
                "from paddle.fluid.param_attr import ParamAttr",
                "import paddle.fluid as fluid", "import math", "",
                "def x2paddle_net():"
            ],
            indent=0)
        for layer_id, layer in self.layers.items():
            edges_in = self.edges_in.get(layer_id, [])
            edges_out = self.edges_out.get(layer_id, [])
            if len(edges_in) == 0 and len(edges_out) == 0:
                continue

            line = ""

            if len(layer.outputs) == 1:
                line = layer.outputs[0]
            else:
                for output in layer.outputs:
                    line += "{}, ".format(output)
                line = line.strip(", ")

            line += " = {}(".format(layer.kernel)
            for k, v in layer.inputs.items():
                if isinstance(v, list):
                    line += "{}=[{}], ".format(k, ", ".join(v))
                else:
                    line += "{}={}, ".format(k, v)
            for k, v in layer.attrs.items():
                line += "{}={}, ".format(k, v)
            line = line.strip(", ")
            line += ")"
            write_code(f, [line], indent=1)

        write_code(
            f, [
                "return [{}], [{}]".format(", ".join(self.inputs),
                                           ", ".join(self.outputs))
            ],
            indent=1)
        f.close()

    def gen_model(self, save_dir, input_shapes=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.graph_type == "static":
            code_dir = os.path.join(save_dir, 'model_with_code')
            infer_dir = os.path.join(save_dir, 'inference_model')
            self.gen_code(code_dir)
            sys.path.append(code_dir)
            import x2paddle_model
            scope = fluid.Scope()
            startup_program = fluid.Program()
            main_program = fluid.Program()
            with fluid.scope_guard(scope):
                with fluid.program_guard(main_program, startup_program):
                    inputs, outputs = x2paddle_model.x2paddle_net()
                    exe = fluid.Executor(fluid.CPUPlace())
                    exe.run(startup_program)

                    param_dir = os.path.join(code_dir, 'weights')
                    for k, v in self.parameters.items():
                        if scope.find_var(k):
                            self.dump_parameter(k, v, param_dir)

                    def if_exist(var):
                        b = os.path.exists(
                            os.path.join(os.path.join(param_dir, var.name)))
                        return b

                    fluid.io.load_vars(
                        exe, param_dir, main_program, predicate=if_exist)
                    fluid.io.save_inference_model(
                        dirname=infer_dir,
                        feeded_var_names=[i.name for i in inputs],
                        target_vars=outputs,
                        executor=exe)
        else:
            self.gen_dygraph_code(save_dir)
            self.dump_dygraph_parameter(save_dir)
            if input_shapes is not None:
                # 如果input_shapes非空，则导出推理模型；其值类似[[None, 3, 224, 224]]
                self.dygraph2static(save_dir, input_shapes)

    def dump_parameter(self, param_name, param, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
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
        if str(param.dtype) in ['uint8', 'uint_8', 'bool']:
            param = param.astype('int64')
        if len(shape) == 0:
            assert param.size == 1, "Unexpected situation happend!"
            shape = [1]
        assert str(
            param.dtype) in dtype_map, "Unknown dtype {} of params: {}.".format(
                str(param.dtype), param_name)
        fp = open(os.path.join(save_dir, param_name), 'wb')
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

    def get_dygraph_inputs(self):
        def update(layers):
            for layer_id, layer in layers.items():
                if self.edges_in.get(layer_id, 0) == 0 and self.edges_out.get(
                        layer_id, 0) == 0:
                    continue
                if layer.kernel == "fluid.dygraph.base.to_variable":
                    value = layer.attrs["value"]
                    if not value.startswith("params["):
                        self.inputs.append(value)
                if len(layer.blocks) > 0:
                    for block in layer.blocks:
                        block.get_dygraph_inputs()
                        self.inputs.extend(block.inputs)

        update(self.layers)
        self.inputs = list(set(self.inputs))
        if self.inputs is not None:
            self.inputs.sort()

    def get_dygraph_outputs(self):
        for layer_id, layer in self.layers.items():
            if self.edges_in.get(layer_id, 0) == 0 and self.edges_out.get(
                    layer_id, 0) == 0:
                continue
            if self.edges_out.get(layer_id, 0) == 0:
                for output_name in layer.outputs:
                    if not output_name.startswith("x"):
                        continue
                    self.outputs.append(output_name)
        self.outputs = list(set(self.outputs))

    def gen_dygraph_code(self, code_dir=None, indent=2):
        def gen_codes(code_list, indent=0):
            indent_blank = "    " * indent
            codes = []
            for code_line in code_list:
                if code_line.strip() == "":
                    codes.append('\n')
                else:
                    codes.append(indent_blank + code_line + '\n')
            return codes

        def gen_head():
            self.head = gen_codes(
                [
                    "from paddle.fluid.initializer import Constant",
                    "from paddle.fluid.param_attr import ParamAttr",
                    "import paddle",
                    "import paddle.fluid as fluid",
                    "",
                    "class {}(fluid.dygraph.Layer):".format(self.name),
                ],
                indent=0)
            input_data_name = ', '.join(self.inputs)
            self.init_func.extend(
                gen_codes(
                    ["def __init__(self, params):"], indent=1))
            self.init_func.extend(
                gen_codes(
                    ["super({}, self).__init__()".format(self.name)], indent=2))
            self.forward_func.extend(
                gen_codes(
                    ["def forward(self, {}):".format(input_data_name)],
                    indent=1))

        def write_code(code_dir):
            f = open(os.path.join(code_dir, 'x2paddle_code.py'), 'w')
            for code_line in self.head:
                f.write(code_line)
            init_writen_codes = []
            for code_line in self.init_func:
                if code_line in init_writen_codes:
                    continue
                f.write(code_line)
                init_writen_codes.append(code_line)
            f.write("\n")
            return_code = "return {}".format(", ".join(self.outputs))
            self.forward_func.extend(gen_codes([return_code], indent=2))
            for code_line in self.forward_func:
                f.write(code_line)
            f.close()

        self.init_func = []
        self.forward_func = []
        if indent == 2 and code_dir is not None:
            gen_head()

        for layer_id, layer in self.layers.items():
            if ("paddle.nn" in layer.kernel and "functional" not in layer.kernel
                ) or layer.kernel == "fluid.dygraph.base.to_variable" or \
               "paddle.fluid.dygraph" in layer.kernel:
                line = "{}".format(
                    layer.outputs[0]
                ) if layer.kernel == "fluid.dygraph.base.to_variable" and not layer.attrs[
                    "value"].startswith("params[") else "self.{}".format(
                        layer.outputs[0])
                line += " = {}(".format(layer.kernel)
                for k, v in layer.attrs.items():
                    line += "{}={}, ".format(k, v)
                line = line.strip(", ")
                line += ")"

                if layer.kernel == "fluid.dygraph.base.to_variable" and not layer.attrs[
                        "value"].startswith("params["):
                    self.forward_func.extend(gen_codes([line], indent=indent))
                    continue
                else:
                    self.init_func.extend(gen_codes([line], indent=2))

                if len(layer.outputs) == 1:
                    line = layer.outputs[0]
                elif len(layer.outputs) == 2:
                    line = layer.outputs[1]
                else:
                    line = ','.join(layer.outputs[1:])
                if layer.kernel == "fluid.dygraph.base.to_variable" and layer.attrs[
                        "value"].startswith("params["):
                    line += " = self.{}".format(layer.outputs[0])
                else:
                    line += " = self.{}(".format(layer.outputs[0])
                    for k, v in layer.inputs.items():
                        line += "{}, ".format(v)
                    line = line.strip(", ")
                    line += ")"
                self.forward_func.extend(gen_codes([line], indent=indent))
            elif "prim" in layer.kernel:
                func_name = layer.kernel.replace(".", "_")
                from x2paddle.op_mapper.pytorch2paddle import prim2code
                if hasattr(prim2code, func_name):
                    func = getattr(prim2code, func_name)
                    func(
                        layer,
                        indent=indent,
                        init_func=self.init_func,
                        forward_func=self.forward_func)
                else:
                    raise Exception(
                        "The kind {} in paddle model is not supported yet.".
                        format(layer.kernel))
            else:
                if len(layer.outputs) == 1:
                    line = layer.outputs[0]
                else:
                    line = ','.join(layer.outputs)
                line += " = {}(".format(layer.kernel)
                for k, v in layer.inputs.items():
                    line += "{}={}, ".format(k, v)
                for k, v in layer.attrs.items():
                    line += "{}={}, ".format(k, v)
                line = line.strip(", ")
                line += ")"
                self.forward_func.extend(gen_codes([line], indent=indent))
        if indent == 2:
            write_code(code_dir)
        else:
            return self.init_func, self.forward_func

    def dump_dygraph_parameter(self, code_dir):
        params_output = open(os.path.join(code_dir, 'model.pdparams'), 'wb')
        pickle.dump(self.parameters, params_output)
        params_output.close()

    def dygraph2static(self, save_dir, input_shapes=[]):
        from paddle.fluid.dygraph.jit import declarative
        sepc_list = list()
        for i, name in enumerate(self.inputs):
            sepc_list.append(
                paddle.static.InputSpec(
                    shape=input_shapes[i], name=name))
        import sys
        path = osp.abspath(save_dir)
        sys.path.insert(0, save_dir)
        import x2paddle_code
        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            restore, _ = fluid.load_dygraph(osp.join(save_dir, "model"))
            model = getattr(x2paddle_code, self.name)(restore)
            model.set_dict(restore)
            model.eval()
            model.forward = declarative(model.forward, sepc_list)
        fluid.dygraph.jit.save(
            layer=model, model_path=osp.join(save_dir, "inference"))
