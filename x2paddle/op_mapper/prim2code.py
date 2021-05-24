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

NO_OUTPUT_COUNT = 0


def gen_codes(code_list, indent=0):
    indent_blank = "    " * indent
    codes = []
    for code_line in code_list:
        if code_line.strip() == "":
            codes.append('\n')
        else:
            codes.append(indent_blank + code_line + '\n')
    return codes


def get_value(layer, key, layer_id=None, different_attrs=None):
    """ 进行optimizer后可能把inputs的value直接用数值代替（ConstantFuser），
        会把input换成attr，所以需要此处的操作。
    """
    if key in layer.inputs:
        return layer.inputs[key]
    else:
        if different_attrs is None:
            return str(layer.attrs[key])
        else:
            key_name = "{}_{}".format(layer.outputs[0], key)
            if key_name in different_attrs:
                return key_name
            else:
                if layer_id is None:
                    return str(layer.attrs[key])
                key_name = "{}_{}".format("layer_id/{}".format(layer_id), key)
                if key_name in different_attrs:
                    new_key_name = "attr_{}".format(NO_OUTPUT_COUNT)
                    NO_OUTPUT_COUNT += 1
                    diff_index = different_attrs.index(key_name)
                    different_attrs[diff_index] = new_key_name
                    return new_key_name
                else:
                    return str(layer.attrs[key])


def prim_add(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = {} + {}".format(layer.outputs[0],
                                 get_value(layer, "x", different_attrs),
                                 get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_add_(layer,
              indent=1,
              init_func=[],
              forward_func=[],
              layer_id=None,
              different_attrs=None):
    line = "{} = {} + {} * {}".format(layer.outputs[0],
                                      get_value(layer, "x", different_attrs),
                                      layer.attrs["alpha"],
                                      get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_and(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = {} and {}".format(layer.outputs[0],
                                   get_value(layer, "x", different_attrs),
                                   get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_append(layer,
                indent=1,
                init_func=[],
                forward_func=[],
                layer_id=None,
                different_attrs=None):
    line = "{}.append({})".format(
        get_value(layer, "list", layer_id, different_attrs),
        get_value(layer, "element", layer_id, different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_assert(layer,
                indent=1,
                init_func=[],
                forward_func=[],
                layer_id=None,
                different_attrs=None):
    if layer.attrs["type"] == "eq":
        values = get_value(layer, "key")
        if "value" in layer.attrs:
            values = layer.attrs["value"]
        if isinstance(values, list):
            s = ""
            for v in values:
                s += "{} == {} or ".format(get_value(layer, "key"), v)
            if len(s) > 0:
                s = s[:-4]
            line = "assert {}, \'The {} must be {}!\'".format(
                s, get_value(layer, "key"), get_value(layer, "value"))
        else:
            line = "assert {} == {}, \'The {} must be {}!\'".format(
                get_value(layer, "key"),
                get_value(layer, "value"),
                get_value(layer, "key"), get_value(layer, "value"))
    else:
        raise Exception("Not implement yet!")
    forward_func.extend(gen_codes([line], indent=indent))


def prim_check_dim(layer,
                   indent=1,
                   init_func=[],
                   forward_func=[],
                   layer_id=None,
                   different_attrs=None):
    lines = []
    dim = get_value(layer, "dim", different_attrs)
    lines.append("if {} < 0:".format(dim))
    lines.append("    {} = {} + {}".format(layer.outputs[
        0], dim, get_value(layer, "len", different_attrs)))
    lines.append("else:")
    lines.append("    {} = {}".format(layer.outputs[0], dim))
    forward_func.extend(gen_codes(lines, indent=indent))


def prim_constant(layer,
                  indent=1,
                  init_func=[],
                  forward_func=[],
                  layer_id=None,
                  different_attrs=None):
    line = "{} = {}".format(layer.outputs[0], layer.attrs["value"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_contain(layer,
                 indent=1,
                 init_func=[],
                 forward_func=[],
                 layer_id=None,
                 different_attrs=None):
    line = "{} = {} in {}".format(layer.outputs[0],
                                  get_value(layer, "element", different_attrs),
                                  get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_dict(layer,
              indent=1,
              init_func=[],
              forward_func=[],
              layer_id=None,
              different_attrs=None):
    line = "{} = dict()".format(layer.outputs[0])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_dict_construct(layer,
                        indent=1,
                        init_func=[],
                        forward_func=[],
                        layer_id=None,
                        different_attrs=None):
    lines = list()
    line = "{} = dict()".format(layer.outputs[0])
    lines.append(line)
    for i in range(len(layer.inputs)):
        line = "{}[{}] = {}".format(
            layer.outputs[0],
            get_value(layer, "key{}".format(i), different_attrs),
            get_value(layer, "value{}".format(i), different_attrs))
        lines.append(line)
    forward_func.extend(gen_codes(lines, indent=indent))


def prim_div(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = {} / {}".format(layer.outputs[0],
                                 get_value(layer, "x", different_attrs),
                                 get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_eq(layer,
            indent=1,
            init_func=[],
            forward_func=[],
            layer_id=None,
            different_attrs=None):
    line = "{} = {} == {}".format(layer.outputs[0],
                                  get_value(layer, "x", different_attrs),
                                  get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_equal(layer,
               indent=1,
               init_func=[],
               forward_func=[],
               layer_id=None,
               different_attrs=None):
    line = "{} = {}".format(layer.outputs[0],
                            get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_exception(layer,
                   indent=1,
                   init_func=[],
                   forward_func=[],
                   layer_id=None,
                   different_attrs=None):
    line = "raise RaiseException({})".format(
        get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_float(layer,
               indent=1,
               init_func=[],
               forward_func=[],
               layer_id=None,
               different_attrs=None):
    line = "{} = float({})".format(layer.outputs[0],
                                   get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_floor(layer,
               indent=1,
               init_func=[],
               forward_func=[],
               layer_id=None,
               different_attrs=None):
    line = "{} = math.floor({})".format(
        layer.outputs[0], get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_floordiv(layer,
                  indent=1,
                  init_func=[],
                  forward_func=[],
                  layer_id=None,
                  different_attrs=None):
    line = "{} = {} // {}".format(layer.outputs[0],
                                  get_value(layer, "x", different_attrs),
                                  get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_getitem(layer,
                 indent=1,
                 init_func=[],
                 forward_func=[],
                 layer_id=None,
                 different_attrs=None):
    line = "{} = {}[{}]".format(layer.outputs[0],
                                get_value(layer, "list", different_attrs),
                                get_value(layer, "index", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_gt(layer,
            indent=1,
            init_func=[],
            forward_func=[],
            layer_id=None,
            different_attrs=None):
    line = "{} = {} > {}".format(layer.outputs[0],
                                 get_value(layer, "x", different_attrs),
                                 get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_if(layer,
            indent=1,
            init_func=[],
            forward_func=[],
            layer_id=None,
            different_attrs=None):
    line = "if {} :".format(get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))
    block = layer.blocks[0]
    b_init_lines, b_forward_lines = block.gen_dygraph_code(indent=indent + 1)
    init_func.extend(b_init_lines)
    forward_func.extend(b_forward_lines)
    block = layer.blocks[1]
    if len(block.layers) > 0:
        b_init_lines, b_forward_lines = block.gen_dygraph_code(
            indent=indent + 1)
        if len(b_forward_lines) != 0:
            line = "else:"
            forward_func.extend(gen_codes([line], indent=indent))
        init_func.extend(b_init_lines)
        forward_func.extend(b_forward_lines)


def prim_int(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = int({})".format(layer.outputs[0],
                                 get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_is(layer,
            indent=1,
            init_func=[],
            forward_func=[],
            layer_id=None,
            different_attrs=None):
    line = "{} = {} is {}".format(layer.outputs[0],
                                  get_value(layer, "x", different_attrs),
                                  get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_isinstance(layer,
                    indent=1,
                    init_func=[],
                    forward_func=[],
                    layer_id=None,
                    different_attrs=None):
    line = "{} = isinstance({}, {})".format(
        layer.outputs[0],
        get_value(layer, "input", different_attrs), layer.attrs["cls"])
    forward_func.extend(gen_codes([line], indent=indent))


def prim_isnot(layer,
               indent=1,
               init_func=[],
               forward_func=[],
               layer_id=None,
               different_attrs=None):
    line = "{} = {} is not {}".format(layer.outputs[0],
                                      get_value(layer, "x", different_attrs),
                                      get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_le(layer,
            indent=1,
            init_func=[],
            forward_func=[],
            layer_id=None,
            different_attrs=None):
    line = "{} = {} <= {}".format(layer.outputs[0],
                                  get_value(layer, "x", different_attrs),
                                  get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_len(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = len({})".format(layer.outputs[0],
                                 get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_len2list(layer,
                  indent=1,
                  init_func=[],
                  forward_func=[],
                  layer_id=None,
                  different_attrs=None):
    lines = []
    lines.append("{} = []".format(layer.outputs[0]))
    lines.append("for i in range({}):".format(
        get_value(layer, "len", different_attrs)))
    lines.append("    {}.append(i)".format(layer.outputs[0]))
    forward_func.extend(gen_codes(lines, indent=indent))


def prim_lt(layer,
            indent=1,
            init_func=[],
            forward_func=[],
            layer_id=None,
            different_attrs=None):
    line = "{} = {} < {}".format(layer.outputs[0],
                                 get_value(layer, "x", different_attrs),
                                 get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_list(layer,
              indent=1,
              init_func=[],
              forward_func=[],
              layer_id=None,
              different_attrs=None):
    input_len = len(layer.inputs) + len(layer.attrs)
    inputs_list = list()
    for i in range(input_len):
        inputs_list.append(
            get_value(layer, "input{}".format(i), different_attrs))
    inputs_str = ', '.join(inputs_list)
    line = "{} = [{}]".format(layer.outputs[0], inputs_str)
    forward_func.extend(gen_codes([line], indent=indent))


def prim_list_unpack(layer,
                     indent=1,
                     init_func=[],
                     forward_func=[],
                     layer_id=None,
                     different_attrs=None):
    line = "{} = {}".format(", ".join(layer.outputs),
                            get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_loop(layer,
              indent=1,
              init_func=[],
              forward_func=[],
              layer_id=None,
              different_attrs=None):
    loop_range = get_value(layer, "input", different_attrs)
    line = "for {} in range({}):".format(layer.outputs[1], loop_range)
    forward_func.extend(gen_codes([line], indent=indent))
    block = layer.blocks[0]
    b_init_lines, b_forward_lines = block.gen_dygraph_code(indent=indent + 1)
    init_func.extend(b_init_lines)
    forward_func.extend(b_forward_lines)


def prim_min(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = min({})".format(layer.outputs[0],
                                 get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_mul(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = {} * {}".format(layer.outputs[0],
                                 get_value(layer, "x", different_attrs),
                                 get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_ne(layer,
            indent=1,
            init_func=[],
            forward_func=[],
            layer_id=None,
            different_attrs=None):
    line = "{} = {} != {}".format(layer.outputs[0],
                                  get_value(layer, "x", different_attrs),
                                  get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_neg(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = -{}".format(layer.outputs[0],
                             get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_not(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = not {}".format(layer.outputs[0],
                                get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_or(layer,
            indent=1,
            init_func=[],
            forward_func=[],
            layer_id=None,
            different_attrs=None):
    line = "{} = {} or {}".format(layer.outputs[0],
                                  get_value(layer, "x", different_attrs),
                                  get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_replaceitem(layer,
                     indent=1,
                     init_func=[],
                     forward_func=[],
                     layer_id=None,
                     different_attrs=None):
    line = "{}[{}] = {}".format(
        get_value(layer, "list", layer_id, different_attrs),
        get_value(layer, "index", layer_id, different_attrs),
        get_value(layer, "item", layer_id, different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_requires_grad(layer,
                       indent=1,
                       init_func=[],
                       forward_func=[],
                       layer_id=None,
                       different_attrs=None):
    line = "{} = not {}.stop_gradient".format(
        layer.outputs[0], get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_rsub(layer,
              indent=1,
              init_func=[],
              forward_func=[],
              layer_id=None,
              different_attrs=None):
    line = "{} = {} - {} * {}".format(
        layer.outputs[0],
        get_value(layer, "y", different_attrs),
        get_value(layer, "x", different_attrs),
        get_value(layer, "alpha", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_select(layer,
                indent=1,
                init_func=[],
                forward_func=[],
                layer_id=None,
                different_attrs=None):
    line = "{} = {}[".format(layer.outputs[0],
                             get_value(layer, "input", different_attrs))
    for dim in range(layer.attrs["dim"]):
        line += ":, "
    line += (get_value(layer, "index", different_attrs) + "]")
    forward_func.extend(gen_codes([line], indent=indent))


def prim_set_attr(layer,
                  indent=1,
                  init_func=[],
                  forward_func=[],
                  layer_id=None,
                  different_attrs=None):
    line = "{} = {}".format(layer.outputs[0],
                            get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_set_item(layer,
                  indent=1,
                  init_func=[],
                  forward_func=[],
                  layer_id=None,
                  different_attrs=None):
    line = "{}[{}] = {}".format(
        get_value(layer, "dict", different_attrs),
        get_value(layer, "key", different_attrs),
        get_value(layer, "value", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_shape_dim(layer,
                   indent=1,
                   init_func=[],
                   forward_func=[],
                   layer_id=None,
                   different_attrs=None):
    line = "{} = fluid.layers.shape({})[{}]".format(
        layer.outputs[0],
        get_value(layer, "input", different_attrs),
        get_value(layer, "dim", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_slice(layer,
               indent=1,
               init_func=[],
               forward_func=[],
               layer_id=None,
               different_attrs=None):
    line = "{} = {}[{}: {}: {}]".format(
        layer.outputs[0],
        get_value(layer, "input", different_attrs),
        get_value(layer, "start", different_attrs),
        get_value(layer, "end", different_attrs),
        get_value(layer, "step", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_str(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = str({})".format(layer.outputs[0],
                                 get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_sub(layer,
             indent=1,
             init_func=[],
             forward_func=[],
             layer_id=None,
             different_attrs=None):
    line = "{} = {} - {}".format(layer.outputs[0],
                                 get_value(layer, "x", different_attrs),
                                 get_value(layer, "y", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_tuple(layer,
               indent=1,
               init_func=[],
               forward_func=[],
               layer_id=None,
               different_attrs=None):
    input_len = len(layer.inputs) + len(layer.attrs)
    inputs_list = list()
    for i in range(input_len):
        inputs_list.append(
            get_value(layer, "input{}".format(i), different_attrs))
    inputs_str = ', '.join(inputs_list)
    line = "{} = ({})".format(layer.outputs[0], inputs_str)
    forward_func.extend(gen_codes([line], indent=indent))


def prim_tuple_unpack(layer,
                      indent=1,
                      init_func=[],
                      forward_func=[],
                      layer_id=None,
                      different_attrs=None):
    outputs_str = ', '.join(layer.outputs)
    line = "{} = {}".format(outputs_str,
                            get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_type(layer,
              indent=1,
              init_func=[],
              forward_func=[],
              layer_id=None,
              different_attrs=None):
    line = "{} = {}.dtype".format(layer.outputs[0],
                                  get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_var2list(layer,
                  indent=1,
                  init_func=[],
                  forward_func=[],
                  layer_id=None,
                  different_attrs=None):
    line = "{} = {}.numpy().tolist()".format(
        layer.outputs[0], get_value(layer, "input", different_attrs))
    forward_func.extend(gen_codes([line], indent=indent))


def prim_warnings(layer,
                  indent=1,
                  init_func=[],
                  forward_func=[],
                  layer_id=None,
                  different_attrs=None):
    lines = ["import warnings"]
    line = "warnings.warn({}, stacklevel={})".format(
        get_value(layer, "input", different_attrs), layer.attrs["stacklevel"])
    lines.append(line)
    forward_func.extend(gen_codes(lines, indent=indent))
