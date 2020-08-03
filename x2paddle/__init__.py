__version__ = "0.7.4"

from .core.program import PaddleProgram

program = PaddleProgram()

name_counter = dict()


def gen_name(op_name, var_name):
    name = "{}_{}".format(op_name, var_name)
    if name not in name_counter:
        name_counter[name] = 0
    else:
        name_counter[name] += 1
    name = name + "_" + str(name_counter[name])
    return name
