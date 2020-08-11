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


def convert_prim(layer, indent=1, init_func=[], forward_func=[]):
    def gen_codes(code_list, indent=0):
        indent_blank = "    " * indent
        codes = []
        for code_line in code_list:
            if code_line.strip() == "":
                codes.append('\n')
            else:
                codes.append(indent_blank + code_line + '\n')
        return codes

    if layer.kernel == "prim.if":
        line = "if {} :".format(list(layer.inputs.values())[0])
        forward_func.extend(gen_codes([line], indent=indent))
        block = layer.blocks[0]
        b_init_lines, b_forward_lines = block.gen_dygraph_code(
            indent=indent + 1)
        init_func.extend(b_init_lines)
        forward_func.extend(b_forward_lines)
        block = layer.blocks[1]
        if len(block.layers) > 0:
            line = "else:"
            forward_func.extend(gen_codes([line], indent=indent))
            b_init_lines, b_forward_lines = block.gen_dygraph_code(
                indent=indent + 1)
            init_func.extend(b_init_lines)
            forward_func.extend(b_forward_lines)
        return
    elif layer.kernel == "prim.loop":
        loop_range = list(layer.inputs.values())[0]
        if list(layer.inputs.values())[0] is None:
            loop_range = str(layer.attrs[list(layer.inputs.keys())[0]])
        line = "for {} in range({}):".format(layer.outputs[1], loop_range)
        forward_func.extend(gen_codes([line], indent=indent))
        block = layer.blocks[0]
        b_init_lines, b_forward_lines = block.gen_dygraph_code(
            indent=indent + 1)
        init_func.extend(b_init_lines)
        forward_func.extend(b_forward_lines)
        return
    elif layer.kernel == "prim.equal":
        line = "{} = {}".format(layer.outputs[0],
                                list(layer.inputs.values())[0])
    elif layer.kernel == "prim.constant":
        line = "{} = {}".format(layer.outputs[0], layer.attrs["value"])
    elif layer.kernel == "prim.list":
        inputs_list = list(layer.inputs.values())
        for i, input in enumerate(inputs_list):
            if input is None:
                inputs_list[i] = str(layer.attrs[list(layer.inputs.keys())[i]])
        inputs_str = ', '.join(inputs_list)
        line = "{} = [{}]".format(layer.outputs[0], inputs_str)
    elif layer.kernel == "prim.exception":
        exception = list(layer.inputs.values())[0]
        if list(layer.inputs.values())[0] is None:
            exception = str(layer.attrs[list(layer.inputs.keys())[0]])
        line = "raise RaiseException({})".format(exception)
    elif layer.kernel == "prim.min":
        line = "{} = min({})".format(layer.outputs[0],
                                     list(layer.inputs.values())[0])
    elif layer.kernel == "prim.add":
        line = "{} = {} + {} * {}".format(layer.outputs[0],
                                          list(layer.inputs.values())[0],
                                          layer.attrs["alpha"],
                                          list(layer.inputs.values())[1])
    elif layer.kernel == "prim.append":
        line = "{} = {}.append({})".format(layer.outputs[0],
                                           list(layer.inputs.values())[0],
                                           list(layer.inputs.values())[1])
    elif layer.kernel == "prim.shape":
        line = "{} = {}.shape".format(layer.outputs[0],
                                      list(layer.inputs.values())[0])
    elif layer.kernel == "prim.len":
        line = "{} = len({})".format(layer.outputs[0],
                                     list(layer.inputs.values())[0])
    elif layer.kernel == "prim.eq":
        line = "{} = {} == {}".format(layer.outputs[0],
                                      list(layer.inputs.values())[0],
                                      list(layer.inputs.values())[1])
    elif layer.kernel == "prim.assert":
        if layer.attrs["type"] == "eq":
            if isinstance(layer.attrs["value"], list):
                s = ""
                for v in layer.attrs["value"]:
                    s += "{} == {} or ".format(layer.attrs["key"], v)
                if len(s) > 0:
                    s = s[:-4]
                line = "assert {}, \'The {} must be {}!\'".format(
                    s, layer.attrs["key"], layer.attrs["value"])
            else:
                line = "assert {} == {}, \'The {} must be {}!\'".format(
                    layer.attrs["key"], layer.attrs["value"],
                    layer.attrs["key"], layer.attrs["value"])
        else:
            raise Exception("Not implement yet!")
    elif layer.kernel == "prim.getitem":
        item0 = list(layer.inputs.values())[0]
        if list(layer.inputs.values())[0] is None:
            item0 = str(layer.attrs[list(layer.inputs.keys())[0]])
        item1 = list(layer.inputs.values())[1]
        if list(layer.inputs.values())[1] is None:
            item1 = str(layer.attrs[list(layer.inputs.keys())[1]])
        line = "{} = {}[{}]".format(layer.outputs[0], item0, item1)
    elif layer.kernel == "prim.le":
        item0 = list(layer.inputs.values())[0]
        if list(layer.inputs.values())[0] is None:
            item0 = str(layer.attrs[list(layer.inputs.keys())[0]])
        item1 = list(layer.inputs.values())[1]
        if list(layer.inputs.values())[1] is None:
            item1 = str(layer.attrs[list(layer.inputs.keys())[1]])
        line = "{} = {} < {}".format(layer.outputs[0], item0, item1)
    elif layer.kernel == "prim.slice":
        attrs_str = ""
        for k, v in layer.attrs.items():
            attrs_str += "{}:".format(v)
        attrs_str = attrs_str[:-1]
        line = "{} = {}[{}]".format(layer.outputs[0],
                                    list(layer.inputs.values())[0], attrs_str)
    forward_func.extend(gen_codes([line], indent=indent))
