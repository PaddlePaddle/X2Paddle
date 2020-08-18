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


def gen_codes(code_list, indent=0):
    indent_blank = "    " * indent
    codes = []
    for code_line in code_list:
        if code_line.strip() == "":
            codes.append('\n')
        else:
            codes.append(indent_blank + code_line + '\n')
    return codes


def prim_add(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} + {}".format(layer.outputs[0], layer.inputs["x"],
                                 layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_add_(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} + {} * {}".format(layer.outputs[0], layer.inputs["x"],
                                      layer.attrs["alpha"], layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_and(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} and {}".format(layer.outputs[0], layer.inputs["x"],
                                   layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_append(layer, indent=1, init_func=[], forward_func=[]):
    line = "{}.append({})".format(layer.inputs["list"], layer.inputs["element"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_assert(layer, indent=1, init_func=[], forward_func=[]):
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
                layer.attrs["key"], layer.attrs["value"], layer.attrs["key"],
                layer.attrs["value"])
    else:
        raise Exception("Not implement yet!")
    forward_func.extend(gen_codes([line], indent=indent))


def prim_constant(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {}".format(layer.outputs[0], layer.attrs["value"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_eq(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} == {}".format(layer.outputs[0], layer.inputs["x"],
                                  layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_equal(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {}".format(layer.outputs[0], layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_exception(layer, indent=1, init_func=[], forward_func=[]):
    line = "raise RaiseException({})".format(layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_if(layer, indent=1, init_func=[], forward_func=[]):
    line = "if {} :".format(list(layer.inputs.values())[0])
    forward_func.extend(gen_codes([line], indent=indent))
    block = layer.blocks[0]
    b_init_lines, b_forward_lines = block.gen_dygraph_code(indent=indent + 1)
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


def prim_getitem(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {}[{}]".format(layer.outputs[0], layer.inputs["list"],
                                layer.inputs["index"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_gt(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} > {}".format(layer.outputs[0], layer.inputs["x"],
                                 layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_le(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} <= {}".format(layer.outputs[0], layer.inputs["x"],
                                  layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_len(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = len({})".format(layer.outputs[0], layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_lt(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} < {}".format(layer.outputs[0], layer.inputs["x"],
                                 layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_list(layer, indent=1, init_func=[], forward_func=[]):
    inputs_list = list(layer.inputs.values())
    inputs_str = ', '.join(inputs_list)
    line = "{} = [{}]".format(layer.outputs[0], inputs_str)
    forward_func.extend(gen_codes([line], indent=indent))


def prim_loop(layer, indent=1, init_func=[], forward_func=[]):
    loop_range = list(layer.inputs.values())[0]
    if list(layer.inputs.values())[0] is None:
        loop_range = str(layer.attrs[list(layer.inputs.keys())[0]])
    line = "for {} in range({}):".format(layer.outputs[1], loop_range)
    forward_func.extend(gen_codes([line], indent=indent))
    block = layer.blocks[0]
    b_init_lines, b_forward_lines = block.gen_dygraph_code(indent=indent + 1)
    init_func.extend(b_init_lines)
    forward_func.extend(b_forward_lines)


def prim_min(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = min({})".format(layer.outputs[0], layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_mul(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} * {}".format(layer.outputs[0], layer.inputs["x"],
                                 layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_ne(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} < {}".format(layer.outputs[0], layer.inputs["x"],
                                 layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_neg(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = -{}".format(layer.outputs[0], layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_not(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = not {}".format(layer.outputs[0], layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_requires_grad(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = not {}.stop_gradient".format(layer.outputs[0],
                                              layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_select(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {}[".format(layer.outputs[0], layer.inputs["input"])
    for dim in range(layer.attrs["dim"]):
        line += ":, "
    line += (layer.inputs["index"] + "]")
    forward_func.extend(gen_codes([line], indent=indent))


def prim_shape(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {}.shape".format(layer.outputs[0], layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_slice(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {}[{}: {}: {}]".format(
        layer.outputs[0], layer.inputs["input"], layer.inputs["start"],
        layer.inputs["end"], layer.inputs["step"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_sub(layer, indent=1, init_func=[], forward_func=[]):
    line = "{} = {} - {}".format(layer.outputs[0], layer.inputs["x"],
                                 layer.inputs["y"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_tuple(layer, indent=1, init_func=[], forward_func=[]):
    inputs_list = list(layer.inputs.values())
    inputs_str = ', '.join(inputs_list)
    line = "{} = ({})".format(layer.outputs[0], inputs_str)
    forward_func.extend(gen_codes([line], indent=indent))


def prim_tuple_unpack(layer, indent=1, init_func=[], forward_func=[]):
    outputs_str = ', '.join(layer.outputs)
    line = "{} = {}".format(outputs_str, layer.inputs["input"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_warnings(layer, indent=1, init_func=[], forward_func=[]):
    lines = ["import warnings"]
    line = "warnings.warn({}, stacklevel={})".format(layer.inputs["input"],
                                                     layer.attrs["stacklevel"])
    lines.append(line)
    forward_func.extend(gen_codes(lines, indent=indent))
