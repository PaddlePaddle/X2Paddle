# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from collections import defaultdict
import warnings

import paddle
from paddle.regularizer import L2Decay


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def update_parameters(parameters, lr, weight_decay):
    parameters_list = list()
    if parameters is not None:
        for items in parameters:
            if isinstance(items, dict):
                params = items["params"]
                if "lr" in items:
                    for p in params:
                        p.optimize_attr["learning_rate"] = items[
                            "lr"] / lr * p.optimize_attr["learning_rate"]
                if "weight_decay" in items:
                    for p in params:
                        if isinstance(items["weight_decay"], (float, int)):
                            p.regularizer = L2Decay(items["weight_decay"])
                        else:
                            p.regularizer = weight_decay
                for p in params:
                    print(p.regularizer)
                parameters_list.extend(params)
            else:
                parameters_list.append(items)
    return parameters_list


class Momentum(paddle.optimizer.Momentum):
    def __init__(self,
                 params,
                 lr=0.001,
                 momentum=0.0,
                 dampening=0,
                 weight_decay=0.0,
                 nesterov=False):
        assert dampening == 0, "The dampening must be 0 in Momentum!"
        parameters_list = update_parameters(params, lr, weight_decay)
        super().__init__(
            learning_rate=lr,
            momentum=momentum,
            parameters=parameters_list,
            use_nesterov=nesterov,
            weight_decay=weight_decay,
            grad_clip=None,
            name=None)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov)
        self.defaults = defaults

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, paddle.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, paddle.Tensor):
                raise TypeError("optimizer can only optimize Tensors.")
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; ",
                stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def zero_grad(self):
        return self.clear_grad()


class Adam(paddle.optimizer.Adam):
    def __init__(self,
                 params,
                 lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-08,
                 weight_decay=0,
                 amsgrad=False):
        parameters_list = update_parameters(params, lr, weight_decay)
        if weight_decay == 0:
            weight_decay = None
        super().__init__(
            learning_rate=lr,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=eps,
            parameters=parameters_list,
            weight_decay=weight_decay,
            grad_clip=None,
            name=None,
            lazy_mode=False)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        self.defaults = defaults

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(parameters_list)
        if len(param_groups) == 0:
            print(param_groups)
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, paddle.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, paddle.Tensor):
                raise TypeError("optimizer can only optimize Tensors.")
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; ",
                stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def zero_grad(self):
        return self.clear_grad()
