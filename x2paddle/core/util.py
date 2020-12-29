# -*- coding:UTF-8 -*-
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import numpy
import math
import os
import inspect


def string(param):
    """ 生成字符串。
    """
    return "\'{}\'".format(param)

def name_generator(nn_name, nn_name2id):
    """ 生成paddle.nn类op的名字。
    
    Args:
        nn_name (str): 名字。
        nn_name2id (dict): key为名字，value为名字出现的次数-1。
    """
    if nn_name in nn_name2id:
        nn_name2id[nn_name] += 1
    else:
        nn_name2id[nn_name] = 0
    real_nn_name = nn_name + str(nn_name2id[nn_name])
    return real_nn_name

def remove_default_attrs(kernel, attrs):
    """ 删除每个OP的默认参数。
    
    Args:
        kernel (str): OP的类型名字。
        attrs (dict): 目前该OP所包含的参数， key为参数名，value为参数值。
    """
    def get_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
    is_func = True
    if "paddle.nn" in kernel and "functional"not in kernel:
        is_func = False
    import paddle
    obj = paddle
    for i, part in enumerate(kernel.split(".")):
        if i == 0:
            continue
        obj = getattr(obj, part)
    if is_func:
        func = obj
    else:
        func = obj.__init__ 
    default_attrs = get_default_args(func)
    for default_k, default_v in default_attrs.items():
        if default_k in attrs:
            if (isinstance(attrs[default_k], list) or isinstance(attrs[default_k], tuple)) \
                    and not is_func:
                if len(set(attrs[default_k])) == 1:
                    attrs[default_k] = attrs[default_k][0]
            if default_v == attrs[default_k]:
                attrs.pop(default_k)