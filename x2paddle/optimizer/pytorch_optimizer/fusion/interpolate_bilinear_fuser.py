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

import numpy as np
from x2paddle.optimizer.pytorch_optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class InterpolateBilinearFuser(FuseBase):
    def __init__(self):
        super(InterpolateBilinearFuser, self).__init__(graph_type="dygraph")
        import torch
        torch_version = torch.__version__
        torch_version_part = torch_version.split(".")
        if int(torch_version_part[0]) == 1 and int(torch_version_part[1]) > 5:
            self.version_gt_150 = True
        else:
            self.version_gt_150 = False

    def build_pattern(self):
        """ 描述需要替换的双线性插值图结构。
        interpolate_bilinear层模式python实现代码示例:
            x3016 = fluid.layers.shape(input=x3005)
            x3016 = len(x3016)
            x3017 = x3016 - 2
            x3018 = []
            for _x3020 in range(x3017):
                x3018.append(None)
            x3021 = (x3005, x8, None, None)
            x3022 = fluid.layers.shape(input=x3005)
            x3022 = len(x3022)
            x3023 = x3022 == 3
            if x3023 :
                raise RaiseException('Exception')
                x3024 = None
            else:
                x3026 = fluid.layers.shape(input=x3005)
                x3026 = len(x3026)
                x3027 = x3026 == 4
                if x3027 :
                    x3044, x3045, x3046, x3047 = x3021
                    x3048 = x3045 is None
                    if x3048 :
                        x3051 = x3046 is None
                        x3049 = x3051
                        x3050 = x3045
                    else:
                        x3052 = x3045
                        x3049 = False
                        x3050 = x3052
                    if x3049 :
                        raise RaiseException('Exception')
                    x3055 = x3050 is not None
                    if x3055 :
                        x3058 = x3050
                        x3059 = x3046 is not None
                        x3056 = x3059
                        x3057 = x3058
                    else:
                        x3056 = False
                        x3057 = x3050
                    if x3056 :
                        raise RaiseException('Exception')
                        x3060 = None
                        x3061 = None
                    else:
                        x3060 = x3046
                        x3061 = x3057
                    x3063 = x3060 is not None
                    if x3063 :
                        x3065 = x3060
                        x3066 = len(x3065)
                        x3067 = x3066 != 2
                        if x3067 :
                            raise RaiseException('Exception')
                        x3064 = x3065
                    else:
                        x3064 = x3060
                    x3070 = x3061 is not None
                    if x3070 :
                        x3072 = x3061
                        x3071 = x3072
                    else:
                        x3071 = None
                    if x3070 :
                        x3073 = x3071
                    else:
                        x3074 = x3064 is not None
                        if x3074 :
                            x3076 = x3064
                            x3075 = x3076
                        else:
                            raise RaiseException('Exception')
                            x3075 = None
                        x3078 = x3047 is None
                        if x3078 :
                            x3080 = len(x3075)
                            x3081 = x3080 > 0
                            x3086 = 0
                            for x3083 in range(2147483647):
                                x3087 = x3075[x3086]
                                x3088 = math.floor(x3087)
                                x3089 = x3088 != x3087
                                if x3089 :
                                    x3090 = False
                                    x3091 = x3089
                                else:
                                    x3090 = None
                                    x3091 = None
                                if x3089 :
                                    x3092 = x3090
                                    x3093 = x3091
                                else:
                                    x3092 = True
                                    x3093 = x3089
                                x3094 = x3086 + 1
                                x3095 = x3094 < x3080
                                x3096 = x3095 and x3092
                                x3082 = x3093
                                x3083 = x3094
                            if x3082 :
                                import warnings
                                warnings.warn('The default behavior for interpolate/upsample with float scale_factor will change in 1.6.0 to align with other frameworks/libraries, and use scale_factor directly, instead of relying on the computed output size. If you wish to keep the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details. ', stacklevel=2)
                        x3099 = []
                        for _x3101 in range(2):
                            x3102 = _x3101 + 2
                            x3103 = fluid.layers.shape(x3044)[x3102]
                            x3104 = float(x3103)
                            x3105 = x3075[_x3101]
                            x3106 = x3104 * x3105
                            x3107 = math.floor(x3106)
                            x3099.append(x3107)
                        x3073 = x3099
                    x3108 = x3018[0]
                    x3109 = x3018[1]
                    x3073_isinstance = isinstance(x3073, paddle.fluid.Variable)
                    if x3073_isinstance :
                        x3073 = x3073.numpy().tolist()
                    assert x3108 == x3109, 'The x3108 must be x3109!'
                    x3110 = paddle.nn.functional.interpolate(x=x3005, size=x3073, scale_factor=x3108, align_corners=False, align_mode=0)
                    x3028 = x3110
                else:
                    x3111 = fluid.layers.shape(input=x3005)
                    x3111 = len(x3111)
                    x3112 = x3111 == 5
                    if x3112 :
                        raise RaiseException('Exception')
                    else:
                        raise RaiseException('Exception')
                    x3028 = None
                x3024 = x3028
        """

        def gen_name(id):
            return "x" + str(id)

        if self.version_gt_150:
            self.pattern.add_layer(
                "fluid.layers.shape",
                inputs={"input": "interpolate-input-0"},
                outputs=[gen_name(9)])
            self.pattern.add_layer(
                "prim.len",
                inputs={"input": gen_name(9)},
                outputs=[gen_name(9)])
            self.pattern.add_layer(
                "prim.sub",
                inputs={"x": gen_name(9)},
                outputs=[gen_name(10)],
                y=2)
            self.pattern.add_layer(
                "prim.list", inputs={}, outputs=[gen_name(11)])
            self.pattern.add_layer(
                "prim.loop",
                inputs={"input": gen_name(10)},
                outputs=[gen_name(12.1), gen_name(12.2)])
            loop_layer = self.pattern.layers[list(self.pattern.layers.keys())[
                -1]]
            pattern_block = PaddleGraph(loop_layer, graph_type="dygraph")
            pattern_block.add_layer(
                "prim.append",
                inputs={"list": gen_name(11)},
                outputs=[],
                element=None)
            loop_layer.inputs["input-0"] = gen_name(11)
            loop_layer.add_block(pattern_block)
            self.pattern.add_layer(
                "prim.tuple",
                inputs={
                    "input0": "interpolate-input-0",
                    "input1": "interpolate-input-1",
                },
                outputs=[gen_name(13)],
                input2=None,
                input3=None)
            self.pattern.add_layer(
                "fluid.layers.shape",
                inputs={"input": "interpolate-input-0"},
                outputs=[gen_name(14)])
            self.pattern.add_layer(
                "prim.len",
                inputs={"input": gen_name(14)},
                outputs=[gen_name(14)])
            self.pattern.add_layer(
                "prim.eq",
                inputs={"x": gen_name(14)},
                outputs=[gen_name(15)],
                y=3)
            self.pattern.add_layer(
                "prim.if",
                inputs={"input": gen_name(15)},
                outputs=[gen_name(16)])
            if_layer1 = self.pattern.layers[list(self.pattern.layers.keys())[
                -1]]
            pattern_block = PaddleGraph(if_layer1, graph_type="dygraph")
            pattern_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(17)],
                input="Exception")
            pattern_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(16)], input=None)
            if_layer1.add_block(pattern_block)
            pattern_block = PaddleGraph(if_layer1, graph_type="dygraph")
            pattern_block.add_layer(
                "fluid.layers.shape",
                inputs={"input": "interpolate-input-0"},
                outputs=[gen_name(18)])
            pattern_block.add_layer(
                "prim.len",
                inputs={"input": gen_name(18)},
                outputs=[gen_name(18)])
            pattern_block.add_layer(
                "prim.eq",
                inputs={"x": gen_name(18)},
                outputs=[gen_name(19)],
                y=4)
            pattern_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(19)},
                outputs=[gen_name(20)])
            if_layer2 = pattern_block.layers[list(pattern_block.layers.keys())[
                -1]]
            pattern_block_block = PaddleGraph(if_layer2, graph_type="dygraph")
            pattern_block_block.add_layer(
                "prim.tuple_unpack",
                inputs={"input": gen_name(13)},
                outputs=[
                    gen_name(34), gen_name(35), gen_name(36), gen_name(37)
                ])
            pattern_block_block.add_layer(
                "prim.is",
                inputs={"x": gen_name(35)},
                outputs=[gen_name(38)],
                y=None)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(38)},
                outputs=[gen_name(39), gen_name(40)])
            if_layer3 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer3, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.is",
                inputs={"x": gen_name(36)},
                outputs=[gen_name(41)],
                y=None)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(41)},
                outputs=[gen_name(39)])
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(35)},
                outputs=[gen_name(40)])
            if_layer3.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer3, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(35)},
                outputs=[gen_name(42)])
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(39)], input=False)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(35)},
                outputs=[gen_name(40)])
            if_layer3.add_block(pattern_block_block_block)
            if_layer3.inputs.update({
                "input-0": gen_name(36),
                'input-1': gen_name(35),
                'input-2': gen_name(35),
            })
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(39)},
                outputs=[gen_name(43)])
            if_layer4 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer4, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(44)],
                input="Exception")
            if_layer4.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer4, graph_type="dygraph")
            if_layer4.add_block(pattern_block_block_block)
            pattern_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(40)},
                outputs=[gen_name(45)],
                y=None)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(45)},
                outputs=[gen_name(46), gen_name(47)])
            if_layer5 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer5, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(40)},
                outputs=[gen_name(48)])
            pattern_block_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(36)},
                outputs=[gen_name(49)],
                y=None)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(49)},
                outputs=[gen_name(46)])
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(48)},
                outputs=[gen_name(47)])
            if_layer5.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer5, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(46)], input=False)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(40)},
                outputs=[gen_name(47)])
            if_layer5.add_block(pattern_block_block_block)
            if_layer5.inputs.update({
                "input-0": gen_name(40),
                "input-1": gen_name(36),
                "input-3": gen_name(40)
            })
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(46)},
                outputs=[gen_name(50), gen_name(51)])
            if_layer6 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer6, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(52)],
                input="Exception")
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(50)], input=None)
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(51)], input=None)
            if_layer6.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer6, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(36)},
                outputs=[gen_name(50)])
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(47)},
                outputs=[gen_name(51)])
            if_layer6.add_block(pattern_block_block_block)
            if_layer6.inputs.update({
                "input-0": gen_name(36),
                "input-1": gen_name(47)
            })
            pattern_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(50)},
                outputs=[gen_name(53)],
                y=None)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(53)},
                outputs=[gen_name(54)])
            if_layer7 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer7, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(50)},
                outputs=[gen_name(55)])
            pattern_block_block_block.add_layer(
                "prim.len",
                inputs={"input": gen_name(55)},
                outputs=[gen_name(56)])
            pattern_block_block_block.add_layer(
                "prim.ne",
                inputs={"x": gen_name(56)},
                outputs=[gen_name(57)],
                y=2)
            pattern_block_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(57)},
                outputs=[gen_name(58)])
            if_layer8 = pattern_block_block_block.layers[list(
                pattern_block_block_block.layers.keys())[-1]]
            pattern_block_block_block_block = PaddleGraph(
                if_layer8, graph_type="dygraph")
            pattern_block_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(59)],
                input="Exception")
            if_layer8.add_block(pattern_block_block_block_block)
            pattern_block_block_block_block = PaddleGraph(
                if_layer8, graph_type="dygraph")
            if_layer8.add_block(pattern_block_block_block_block)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(55)},
                outputs=[gen_name(54)])
            if_layer7.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer7, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(50)},
                outputs=[gen_name(54)])
            if_layer7.add_block(pattern_block_block_block)
            if_layer7.inputs.update({
                "input-0": gen_name(50),
                "input-1": gen_name(50)
            })
            pattern_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(51)},
                outputs=[gen_name(60)],
                y=None)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(60)},
                outputs=[gen_name(61)])
            if_layer9 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer9, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(51)},
                outputs=[gen_name(62)])
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(62)},
                outputs=[gen_name(61)])
            if_layer9.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer9, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(54)},
                outputs=[gen_name(64)],
                y=None)
            pattern_block_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(64)},
                outputs=[gen_name(65)])
            if_layer11 = pattern_block_block_block.layers[list(
                pattern_block_block_block.layers.keys())[-1]]
            pattern_block_block_block_block = PaddleGraph(
                if_layer11, graph_type="dygraph")
            pattern_block_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(54)},
                outputs=[gen_name(66)])
            pattern_block_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(66)},
                outputs=[gen_name(65)])
            if_layer11.add_block(pattern_block_block_block_block)
            pattern_block_block_block_block = PaddleGraph(
                if_layer11, graph_type="dygraph")
            pattern_block_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(67)],
                input="Exception")
            pattern_block_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(65)], input=None)
            if_layer11.add_block(pattern_block_block_block_block)
            if_layer11.inputs.update({"input-0": gen_name(54), })
            pattern_block_block_block.add_layer(
                "prim.is",
                inputs={"x": gen_name(37)},
                outputs=[gen_name(68)],
                y=None)
            pattern_block_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(68)},
                outputs=[gen_name(69)])
            if_layer12 = pattern_block_block_block.layers[list(
                pattern_block_block_block.layers.keys())[-1]]
            pattern_block_block_block_block = PaddleGraph(
                if_layer12, graph_type="dygraph")
            pattern_block_block_block_block.add_layer(
                "prim.len",
                inputs={"input": gen_name(65)},
                outputs=[gen_name(70)])
            pattern_block_block_block_block.add_layer(
                "prim.gt",
                inputs={"x": gen_name(70)},
                outputs=[gen_name(71)],
                y=0)
            pattern_block_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(72)], input=0)
            pattern_block_block_block_block.add_layer(
                "prim.loop",
                inputs={},
                outputs=[gen_name(74), gen_name(75), gen_name(76.1)],
                input=2147483647)
            loop_layer = pattern_block_block_block_block.layers[list(
                pattern_block_block_block_block.layers.keys())[-1]]
            pattern_loop_block = PaddleGraph(loop_layer, graph_type="dygraph")
            pattern_loop_block.add_layer(
                "prim.getitem",
                inputs={"list": gen_name(65),
                        "element": gen_name(72)},
                outputs=[gen_name(74.1)])
            pattern_loop_block.add_layer(
                "prim.floor",
                inputs={"input": gen_name(74.1)},
                outputs=[gen_name(75.1)])
            pattern_loop_block.add_layer(
                "prim.ne",
                inputs={"x": gen_name(75.1),
                        "y": gen_name(74.1)},
                outputs=[gen_name(76)])
            pattern_loop_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(76)},
                outputs=[gen_name(77)])
            if_layer13 = pattern_loop_block.layers[list(
                pattern_loop_block.layers.keys())[-1]]
            pattern_loop_block_block = PaddleGraph(
                if_layer13, graph_type="dygraph")
            pattern_loop_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(77)], input=False)
            if_layer13.add_block(pattern_loop_block_block)
            pattern_loop_block_block = PaddleGraph(
                if_layer13, graph_type="dygraph")
            pattern_loop_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(77)], input=True)
            if_layer13.add_block(pattern_loop_block_block)
            pattern_loop_block.add_layer(
                "prim.add",
                inputs={"x": gen_name(72)},
                outputs=[gen_name(81)],
                y=1)
            pattern_loop_block.add_layer(
                "prim.lt",
                inputs={"x": gen_name(81),
                        "y": gen_name(70)},
                outputs=[gen_name(82)])
            pattern_loop_block.add_layer(
                "prim.and",
                inputs={"x": gen_name(82),
                        "y": gen_name(77)},
                outputs=[gen_name(83)])
            pattern_loop_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(76)},
                outputs=[gen_name(74)])
            pattern_loop_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(81)},
                outputs=[gen_name(75)])
            loop_layer.add_block(pattern_loop_block)
            loop_layer.inputs.update({
                "input-0": gen_name(65),
                "input-1": gen_name(72),
                "input-2": gen_name(72),
                "input-3": gen_name(70)
            })
            pattern_block_block_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(74)},
                outputs=[gen_name(84)])
            if_layer15 = pattern_block_block_block_block.layers[list(
                pattern_block_block_block_block.layers.keys())[-1]]
            pattern_block_block_block_block_block = PaddleGraph(
                if_layer15, graph_type="dygraph")
            pattern_block_block_block_block_block.add_layer(
                "prim.warnings",
                inputs={},
                outputs=[gen_name(85)],
                stacklevel=2,
                input="...")
            if_layer15.add_block(pattern_block_block_block_block_block)
            pattern_block_block_block_block_block = PaddleGraph(
                if_layer15, graph_type="dygraph")
            if_layer15.add_block(pattern_block_block_block_block_block)
            if_layer12.add_block(pattern_block_block_block_block)
            pattern_block_block_block_block = PaddleGraph(
                if_layer12, graph_type="dygraph")
            if_layer12.add_block(pattern_block_block_block_block)
            if_layer12.inputs.update({
                "input-0": gen_name(65),
                "input-1": gen_name(65),
            })
            pattern_block_block_block.add_layer(
                "prim.list", inputs={}, outputs=[gen_name(86)])
            pattern_block_block_block.add_layer(
                "prim.loop",
                inputs={},
                outputs=[gen_name(87), gen_name(88)],
                input=2)
            loop_layer = pattern_block_block_block.layers[list(
                pattern_block_block_block.layers.keys())[-1]]
            pattern_loop_block = PaddleGraph(loop_layer, graph_type="dygraph")
            pattern_loop_block.add_layer(
                "prim.add",
                inputs={"x": gen_name(88)},
                outputs=[gen_name(89)],
                y=2)
            pattern_loop_block.add_layer(
                "prim.shape_dim",
                inputs={"input": gen_name(34),
                        "dim": gen_name(89)},
                outputs=[gen_name(90)])
            pattern_loop_block.add_layer(
                "prim.float",
                inputs={"input": gen_name(90)},
                outputs=[gen_name(91)])
            pattern_loop_block.add_layer(
                "prim.getitem",
                inputs={"list": gen_name(65),
                        "element": gen_name(88)},
                outputs=[gen_name(92)])
            pattern_loop_block.add_layer(
                "prim.mul",
                inputs={"x": gen_name(91),
                        "y": gen_name(92)},
                outputs=[gen_name(93)])
            pattern_loop_block.add_layer(
                "prim.floor",
                inputs={"input": gen_name(93)},
                outputs=[gen_name(94)])
            pattern_loop_block.add_layer(
                "prim.append",
                inputs={"list": gen_name(86),
                        "element": gen_name(94)},
                outputs=[])
            loop_layer.add_block(pattern_loop_block)
            loop_layer.inputs.update({
                "input-0": gen_name(34),
                "input-1": gen_name(65),
                "input-2": gen_name(86)
            })
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(86)},
                outputs=[gen_name(61)])
            if_layer9.add_block(pattern_block_block_block)
            if_layer9.inputs.update({
                "input-0": gen_name(51),
                "input-1": gen_name(54),
                "input-2": gen_name(54),
                "input-3": gen_name(37),
                "input-4": gen_name(34)
            })
            pattern_block_block.add_layer(
                "prim.getitem",
                inputs={"list": gen_name(11)},
                outputs=[gen_name(95)],
                element=0)
            pattern_block_block.add_layer(
                "prim.getitem",
                inputs={"list": gen_name(11)},
                outputs=[gen_name(96)],
                element=1)
            pattern_block_block.add_layer(
                "prim.isinstance",
                inputs={"input": gen_name(61)},
                outputs=["interpolate-input-0_isinstance"],
                cls="paddle.fluid.Variable")
            pattern_block_block.add_layer(
                "prim.if", {"input": "interpolate-input-0_isinstance"},
                outputs=["interpolate-input-0_if1"])
            if_layer_isinstance = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer_isinstance, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.var2list",
                inputs={"input": gen_name(61)},
                outputs=[gen_name(61)])
            if_layer_isinstance.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer_isinstance, graph_type="dygraph")
            if_layer_isinstance.add_block(pattern_block_block_block)
            if_layer_isinstance.inputs["input-0"] = gen_name(61)
            pattern_block_block.add_layer(
                "prim.assert",
                inputs={"key": gen_name(95),
                        "value": gen_name(96)},
                outputs=[gen_name(97) + "_assert"],
                type="eq")
            pattern_block_block.add_layer(
                "paddle.nn.functional.interpolate",
                inputs={
                    "input": "interpolate-input-0",
                    "size": gen_name(61),
                    "scale_factor": gen_name(95)
                },
                outputs=[gen_name(97)],
                align_corners=False,
                align_mode=0)
            pattern_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(97)},
                outputs=[gen_name(20)])
            if_layer2.add_block(pattern_block_block)
            pattern_block_block = PaddleGraph(if_layer2, graph_type="dygraph")
            pattern_block_block.add_layer(
                "fluid.layers.shape",
                inputs={"input": "interpolate-input-0"},
                outputs=[gen_name(98)])
            pattern_block_block.add_layer(
                "prim.len",
                inputs={"input": gen_name(98)},
                outputs=[gen_name(98)])
            pattern_block_block.add_layer(
                "prim.eq",
                inputs={"x": gen_name(98)},
                outputs=[gen_name(99)],
                y=5)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(99)},
                outputs=[gen_name(100)])
            if_layer16 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer16, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(101)],
                input="Exception")
            if_layer16.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer16, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(102)],
                input="Exception")
            if_layer16.add_block(pattern_block_block_block)
            pattern_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(20)], input=None)
            if_layer2.add_block(pattern_block_block)
            if_layer2.inputs.update({
                "input-0": gen_name(13),
                "input-1": gen_name(13),
                "input-2": "interpolate-input-0",
                "input-3": gen_name(11),
                "input-5": gen_name(11),
            })
            pattern_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(20)},
                outputs=[gen_name(16)])
            if_layer1.add_block(pattern_block)
            if_layer1.inputs.update({
                "input-2": "interpolate-input-0",
                "input-4": gen_name(13),
                "input-7": gen_name(11),
                "input-9": gen_name(11),
                "input-11": "interpolate-input-0",
                "input-12": "interpolate-input-0",
            })
            self.pattern.build(inputs={
                "input-0": "interpolate-input-0",
                "input-1": "interpolate-input-1"
            })
        else:
            self.pattern.add_layer(
                "fluid.layers.shape",
                inputs={"input": "interpolate-input-0"},
                outputs=[gen_name(9)])
            self.pattern.add_layer(
                "prim.len",
                inputs={"input": gen_name(9)},
                outputs=[gen_name(9)])
            self.pattern.add_layer(
                "prim.sub",
                inputs={"x": gen_name(9)},
                outputs=[gen_name(10)],
                y=2)
            self.pattern.add_layer(
                "prim.list", inputs={}, outputs=[gen_name(11)])
            self.pattern.add_layer(
                "prim.loop",
                inputs={"input": gen_name(10)},
                outputs=[gen_name(12.1), gen_name(12.2)])
            loop_layer = self.pattern.layers[list(self.pattern.layers.keys())[
                -1]]
            pattern_block = PaddleGraph(loop_layer, graph_type="dygraph")
            pattern_block.add_layer(
                "prim.append",
                inputs={"list": gen_name(11)},
                outputs=[],
                element=None)
            loop_layer.inputs["input-0"] = gen_name(11)
            loop_layer.add_block(pattern_block)
            self.pattern.add_layer(
                "prim.tuple",
                inputs={
                    "input0": "interpolate-input-0",
                    "input1": "interpolate-input-1",
                },
                outputs=[gen_name(13)],
                input2=None,
                input3=None)
            self.pattern.add_layer(
                "fluid.layers.shape",
                inputs={"input": "interpolate-input-0"},
                outputs=[gen_name(14)])
            self.pattern.add_layer(
                "prim.len",
                inputs={"input": gen_name(14)},
                outputs=[gen_name(14)])
            self.pattern.add_layer(
                "prim.eq",
                inputs={"x": gen_name(14)},
                outputs=[gen_name(15)],
                y=3)
            self.pattern.add_layer(
                "prim.if",
                inputs={"input": gen_name(15)},
                outputs=[gen_name(16)])
            if_layer1 = self.pattern.layers[list(self.pattern.layers.keys())[
                -1]]
            pattern_block = PaddleGraph(if_layer1, graph_type="dygraph")
            pattern_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(17)],
                input="Exception")
            pattern_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(16)], input=None)
            if_layer1.add_block(pattern_block)
            pattern_block = PaddleGraph(if_layer1, graph_type="dygraph")
            pattern_block.add_layer(
                "fluid.layers.shape",
                inputs={"input": "interpolate-input-0"},
                outputs=[gen_name(18)])
            pattern_block.add_layer(
                "prim.len",
                inputs={"input": gen_name(18)},
                outputs=[gen_name(18)])
            pattern_block.add_layer(
                "prim.eq",
                inputs={"x": gen_name(18)},
                outputs=[gen_name(19)],
                y=4)
            pattern_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(19)},
                outputs=[gen_name(20)])
            if_layer2 = pattern_block.layers[list(pattern_block.layers.keys())[
                -1]]
            pattern_block_block = PaddleGraph(if_layer2, graph_type="dygraph")
            pattern_block_block.add_layer(
                "prim.tuple_unpack",
                inputs={"input": gen_name(13)},
                outputs=[
                    gen_name(34), gen_name(35), gen_name(36), gen_name(37)
                ])
            pattern_block_block.add_layer(
                "prim.is",
                inputs={"x": gen_name(35)},
                outputs=[gen_name(38)],
                y=None)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(38)},
                outputs=[gen_name(39), gen_name(40)])
            if_layer3 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer3, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.is",
                inputs={"x": gen_name(36)},
                outputs=[gen_name(41)],
                y=None)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(41)},
                outputs=[gen_name(39)])
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(35)},
                outputs=[gen_name(40)])
            if_layer3.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer3, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(35)},
                outputs=[gen_name(42)])
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(39)], input=False)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(35)},
                outputs=[gen_name(40)])
            if_layer3.add_block(pattern_block_block_block)
            if_layer3.inputs.update({
                "input-0": gen_name(36),
                'input-1': gen_name(35),
                'input-2': gen_name(35),
            })
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(39)},
                outputs=[gen_name(43)])
            if_layer4 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer4, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(44)],
                input="Exception")
            if_layer4.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer4, graph_type="dygraph")
            if_layer4.add_block(pattern_block_block_block)
            pattern_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(40)},
                outputs=[gen_name(45)],
                y=None)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(45)},
                outputs=[gen_name(46), gen_name(47)])
            if_layer5 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer5, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(40)},
                outputs=[gen_name(48)])
            pattern_block_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(36)},
                outputs=[gen_name(49)],
                y=None)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(49)},
                outputs=[gen_name(46)])
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(48)},
                outputs=[gen_name(47)])
            if_layer5.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer5, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(46)], input=False)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(40)},
                outputs=[gen_name(47)])
            if_layer5.add_block(pattern_block_block_block)
            if_layer5.inputs.update({
                "input-0": gen_name(40),
                "input-1": gen_name(36),
                "input-3": gen_name(40)
            })
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(46)},
                outputs=[gen_name(50), gen_name(51)])
            if_layer6 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer6, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(52)],
                input="Exception")
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(50)], input=None)
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(51)], input=None)
            if_layer6.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer6, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(36)},
                outputs=[gen_name(50)])
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(47)},
                outputs=[gen_name(51)])
            if_layer6.add_block(pattern_block_block_block)
            if_layer6.inputs.update({
                "input-0": gen_name(36),
                "input-1": gen_name(47)
            })
            pattern_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(50)},
                outputs=[gen_name(53)],
                y=None)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(53)},
                outputs=[gen_name(54)])
            if_layer7 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer7, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(50)},
                outputs=[gen_name(55)])
            pattern_block_block_block.add_layer(
                "prim.len",
                inputs={"input": gen_name(55)},
                outputs=[gen_name(56)])
            pattern_block_block_block.add_layer(
                "prim.ne",
                inputs={"x": gen_name(56)},
                outputs=[gen_name(57)],
                y=2)
            pattern_block_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(57)},
                outputs=[gen_name(58)])
            if_layer8 = pattern_block_block_block.layers[list(
                pattern_block_block_block.layers.keys())[-1]]
            pattern_block_block_block_block = PaddleGraph(
                if_layer8, graph_type="dygraph")
            pattern_block_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(59)],
                input="Exception")
            if_layer8.add_block(pattern_block_block_block_block)
            pattern_block_block_block_block = PaddleGraph(
                if_layer8, graph_type="dygraph")
            if_layer8.add_block(pattern_block_block_block_block)
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(55)},
                outputs=[gen_name(54)])
            if_layer7.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer7, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(50)},
                outputs=[gen_name(54)])
            if_layer7.add_block(pattern_block_block_block)
            if_layer7.inputs.update({
                "input-0": gen_name(50),
                "input-1": gen_name(50)
            })
            pattern_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(51)},
                outputs=[gen_name(60)],
                y=None)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(60)},
                outputs=[gen_name(61)])
            if_layer9 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer9, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(51)},
                outputs=[gen_name(62)])
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(62)},
                outputs=[gen_name(61)])
            if_layer9.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer9, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(61)], input=None)
            if_layer9.add_block(pattern_block_block_block)
            if_layer9.inputs.update({"input-0": gen_name(51)})
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(60)},
                outputs=[gen_name(63)])
            if_layer10 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer10, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(61)},
                outputs=[gen_name(63)])
            if_layer10.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer10, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.isnot",
                inputs={"x": gen_name(54)},
                outputs=[gen_name(64)],
                y=None)
            pattern_block_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(64)},
                outputs=[gen_name(65)])
            if_layer11 = pattern_block_block_block.layers[list(
                pattern_block_block_block.layers.keys())[-1]]
            pattern_block_block_block_block = PaddleGraph(
                if_layer11, graph_type="dygraph")
            pattern_block_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(54)},
                outputs=[gen_name(66)])
            pattern_block_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(66)},
                outputs=[gen_name(65)])
            if_layer11.add_block(pattern_block_block_block_block)
            pattern_block_block_block_block = PaddleGraph(
                if_layer11, graph_type="dygraph")
            pattern_block_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(67)],
                input="Exception")
            pattern_block_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(65)], input=None)
            if_layer11.add_block(pattern_block_block_block_block)
            if_layer11.inputs.update({"input-0": gen_name(54), })
            pattern_block_block_block.add_layer(
                "prim.is",
                inputs={"x": gen_name(37)},
                outputs=[gen_name(68)],
                y=None)
            pattern_block_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(68)},
                outputs=[gen_name(69)])
            if_layer12 = pattern_block_block_block.layers[list(
                pattern_block_block_block.layers.keys())[-1]]
            pattern_block_block_block_block = PaddleGraph(
                if_layer12, graph_type="dygraph")
            pattern_block_block_block_block.add_layer(
                "prim.len",
                inputs={"input": gen_name(65)},
                outputs=[gen_name(70)])
            pattern_block_block_block_block.add_layer(
                "prim.gt",
                inputs={"x": gen_name(70)},
                outputs=[gen_name(71)],
                y=0)
            pattern_block_block_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(72)], input=0)
            pattern_block_block_block_block.add_layer(
                "prim.loop",
                inputs={},
                outputs=[gen_name(74), gen_name(75), gen_name(76.1)],
                input=2147483647)
            loop_layer = pattern_block_block_block_block.layers[list(
                pattern_block_block_block_block.layers.keys())[-1]]
            pattern_loop_block = PaddleGraph(loop_layer, graph_type="dygraph")
            pattern_loop_block.add_layer(
                "prim.getitem",
                inputs={"list": gen_name(65),
                        "element": gen_name(72)},
                outputs=[gen_name(74.1)])
            pattern_loop_block.add_layer(
                "prim.floor",
                inputs={"input": gen_name(74.1)},
                outputs=[gen_name(75.1)])
            pattern_loop_block.add_layer(
                "prim.ne",
                inputs={"x": gen_name(75.1),
                        "y": gen_name(74.1)},
                outputs=[gen_name(76)])
            pattern_loop_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(76)},
                outputs=[gen_name(77), gen_name(78)])
            if_layer13 = pattern_loop_block.layers[list(
                pattern_loop_block.layers.keys())[-1]]
            pattern_loop_block_block = PaddleGraph(
                if_layer13, graph_type="dygraph")
            pattern_loop_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(77)], input=False)
            pattern_loop_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(76)},
                outputs=[gen_name(78)])
            if_layer13.add_block(pattern_loop_block_block)
            pattern_loop_block_block = PaddleGraph(
                if_layer13, graph_type="dygraph")
            pattern_loop_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(77)], input=None)
            pattern_loop_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(78)], input=None)
            if_layer13.add_block(pattern_loop_block_block)
            if_layer13.inputs.update({"input-0": gen_name(76), })
            pattern_loop_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(76)},
                outputs=[gen_name(79), gen_name(80)])
            if_layer14 = pattern_loop_block.layers[list(
                pattern_loop_block.layers.keys())[-1]]
            pattern_loop_block_block = PaddleGraph(
                if_layer14, graph_type="dygraph")
            pattern_loop_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(77)},
                outputs=[gen_name(79)])
            pattern_loop_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(78)},
                outputs=[gen_name(80)])
            if_layer14.add_block(pattern_loop_block_block)
            pattern_loop_block_block = PaddleGraph(
                if_layer14, graph_type="dygraph")
            pattern_loop_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(79)], input=True)
            pattern_loop_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(76)},
                outputs=[gen_name(80)])
            if_layer14.add_block(pattern_loop_block_block)
            if_layer14.inputs.update({
                "input-0": gen_name(77),
                "input-1": gen_name(78),
                "input-2": gen_name(76)
            })
            pattern_loop_block.add_layer(
                "prim.add",
                inputs={"x": gen_name(72)},
                outputs=[gen_name(81)],
                y=1)
            pattern_loop_block.add_layer(
                "prim.lt",
                inputs={"x": gen_name(81),
                        "y": gen_name(70)},
                outputs=[gen_name(82)])
            pattern_loop_block.add_layer(
                "prim.and",
                inputs={"x": gen_name(82),
                        "y": gen_name(79)},
                outputs=[gen_name(83)])
            pattern_loop_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(80)},
                outputs=[gen_name(74)])
            pattern_loop_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(81)},
                outputs=[gen_name(75)])
            loop_layer.add_block(pattern_loop_block)
            loop_layer.inputs.update({
                "input-0": gen_name(65),
                "input-1": gen_name(72),
                "input-2": gen_name(72),
                "input-3": gen_name(70)
            })
            pattern_block_block_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(74)},
                outputs=[gen_name(84)])
            if_layer15 = pattern_block_block_block_block.layers[list(
                pattern_block_block_block_block.layers.keys())[-1]]
            pattern_block_block_block_block_block = PaddleGraph(
                if_layer15, graph_type="dygraph")
            pattern_block_block_block_block_block.add_layer(
                "prim.warnings",
                inputs={},
                outputs=[gen_name(85)],
                stacklevel=2,
                input="...")
            if_layer15.add_block(pattern_block_block_block_block_block)
            pattern_block_block_block_block_block = PaddleGraph(
                if_layer15, graph_type="dygraph")
            if_layer15.add_block(pattern_block_block_block_block_block)
            if_layer12.add_block(pattern_block_block_block_block)
            pattern_block_block_block_block = PaddleGraph(
                if_layer12, graph_type="dygraph")
            if_layer12.add_block(pattern_block_block_block_block)
            if_layer12.inputs.update({
                "input-0": gen_name(65),
                "input-1": gen_name(65),
            })
            pattern_block_block_block.add_layer(
                "prim.list", inputs={}, outputs=[gen_name(86)])
            pattern_block_block_block.add_layer(
                "prim.loop",
                inputs={},
                outputs=[gen_name(87), gen_name(88)],
                input=2)
            loop_layer = pattern_block_block_block.layers[list(
                pattern_block_block_block.layers.keys())[-1]]
            pattern_loop_block = PaddleGraph(loop_layer, graph_type="dygraph")
            pattern_loop_block.add_layer(
                "prim.add",
                inputs={"x": gen_name(88)},
                outputs=[gen_name(89)],
                y=2)
            pattern_loop_block.add_layer(
                "prim.shape_dim",
                inputs={"input": gen_name(34),
                        "dim": gen_name(89)},
                outputs=[gen_name(90)])
            pattern_loop_block.add_layer(
                "prim.float",
                inputs={"input": gen_name(90)},
                outputs=[gen_name(91)])
            pattern_loop_block.add_layer(
                "prim.getitem",
                inputs={"list": gen_name(65),
                        "element": gen_name(88)},
                outputs=[gen_name(92)])
            pattern_loop_block.add_layer(
                "prim.mul",
                inputs={"x": gen_name(91),
                        "y": gen_name(92)},
                outputs=[gen_name(93)])
            pattern_loop_block.add_layer(
                "prim.floor",
                inputs={"input": gen_name(93)},
                outputs=[gen_name(94)])
            pattern_loop_block.add_layer(
                "prim.append",
                inputs={"list": gen_name(86),
                        "element": gen_name(94)},
                outputs=[])
            loop_layer.add_block(pattern_loop_block)
            loop_layer.inputs.update({
                "input-0": gen_name(34),
                "input-1": gen_name(65),
                "input-2": gen_name(86)
            })
            pattern_block_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(86)},
                outputs=[gen_name(63)])
            if_layer10.add_block(pattern_block_block_block)
            if_layer10.inputs.update({
                "input-0": gen_name(61),
                "input-1": gen_name(54),
                "input-2": gen_name(54),
                "input-3": gen_name(37),
                "input-4": gen_name(34)
            })
            pattern_block_block.add_layer(
                "prim.getitem",
                inputs={"list": gen_name(11)},
                outputs=[gen_name(95)],
                element=0)
            pattern_block_block.add_layer(
                "prim.getitem",
                inputs={"list": gen_name(11)},
                outputs=[gen_name(96)],
                element=1)
            pattern_block_block.add_layer(
                "prim.isinstance",
                inputs={"input": gen_name(63)},
                outputs=["interpolate-input-0_isinstance"],
                cls="paddle.fluid.Variable")
            pattern_block_block.add_layer(
                "prim.if", {"input": "interpolate-input-0_isinstance"},
                outputs=["interpolate-input-0_if1"])
            if_layer_isinstance = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer_isinstance, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.var2list",
                inputs={"input": gen_name(63)},
                outputs=[gen_name(63)])
            if_layer_isinstance.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer_isinstance, graph_type="dygraph")
            if_layer_isinstance.add_block(pattern_block_block_block)
            if_layer_isinstance.inputs["input-0"] = gen_name(63)
            pattern_block_block.add_layer(
                "prim.assert",
                inputs={"key": gen_name(95),
                        "value": gen_name(96)},
                outputs=[gen_name(97) + "_assert"],
                type="eq")
            pattern_block_block.add_layer(
                "paddle.nn.functional.interpolate",
                inputs={
                    "input": "interpolate-input-0",
                    "size": gen_name(63),
                    "scale_factor": gen_name(95)
                },
                outputs=[gen_name(97)],
                align_corners=False,
                align_mode=0)
            pattern_block_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(97)},
                outputs=[gen_name(20)])
            if_layer2.add_block(pattern_block_block)
            pattern_block_block = PaddleGraph(if_layer2, graph_type="dygraph")
            pattern_block_block.add_layer(
                "fluid.layers.shape",
                inputs={"input": "interpolate-input-0"},
                outputs=[gen_name(98)])
            pattern_block_block.add_layer(
                "prim.len",
                inputs={"input": gen_name(98)},
                outputs=[gen_name(98)])
            pattern_block_block.add_layer(
                "prim.eq",
                inputs={"x": gen_name(98)},
                outputs=[gen_name(99)],
                y=5)
            pattern_block_block.add_layer(
                "prim.if",
                inputs={"input": gen_name(99)},
                outputs=[gen_name(100)])
            if_layer16 = pattern_block_block.layers[list(
                pattern_block_block.layers.keys())[-1]]
            pattern_block_block_block = PaddleGraph(
                if_layer16, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(101)],
                input="Exception")
            if_layer16.add_block(pattern_block_block_block)
            pattern_block_block_block = PaddleGraph(
                if_layer16, graph_type="dygraph")
            pattern_block_block_block.add_layer(
                "prim.exception",
                inputs={},
                outputs=[gen_name(102)],
                input="Exception")
            if_layer16.add_block(pattern_block_block_block)
            pattern_block_block.add_layer(
                "prim.equal", inputs={}, outputs=[gen_name(20)], input=None)
            if_layer2.add_block(pattern_block_block)
            if_layer2.inputs.update({
                "input-0": gen_name(13),
                "input-1": gen_name(13),
                "input-2": "interpolate-input-0",
                "input-3": gen_name(11),
                "input-5": gen_name(11),
            })
            pattern_block.add_layer(
                "prim.equal",
                inputs={"input": gen_name(20)},
                outputs=[gen_name(16)])
            if_layer1.add_block(pattern_block)
            if_layer1.inputs.update({
                "input-2": "interpolate-input-0",
                "input-4": gen_name(13),
                "input-7": gen_name(11),
                "input-9": gen_name(11),
                "input-11": "interpolate-input-0",
                "input-12": "interpolate-input-0",
            })
            self.pattern.build(inputs={
                "input-0": "interpolate-input-0",
                "input-1": "interpolate-input-1"
            })

    def insert_new_layer(self, graph, parameters, matches):
        new_layers = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layers[0]
        matches.pop(new_layer_id)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layers[1]
        block_layer = new_layers[1].blocks[0].layers.pop(
            list(new_layers[1].blocks[0].layers.keys())[-1])
        new_layers[1].blocks[0].layers[new_layer_id + ".0.0"] = block_layer
        matches.pop(new_layer_id)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layers[2]
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers = list()
        layers_id = list(matches.keys())
        layer = matches[layers_id[6]]
        size = layer.inputs["input1"]
        layer = matches[layers_id[92]]
        layer.inputs["input"] = size
        layers.append(layer)
        layer = matches[layers_id[93]]
        block_layer = layer.blocks[0].layers[list(layer.blocks[0].layers.keys())
                                             [0]]
        block_layer.inputs["input"] = size
        block_layer.outputs[0] = size
        layer.inputs["input-0"] = size
        layers.append(layer)
        layer = matches[layers_id[-1]]
        outputs = layer.outputs
        layer = matches[layers_id[96]]
        layer.inputs.pop("scale_factor")
        layer.inputs["size"] = size
        layer.outputs = outputs
        layers.append(layer)
        return layers
