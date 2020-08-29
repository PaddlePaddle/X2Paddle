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
from x2paddle.optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class InterpolateBilinearFuser(FuseBase):
    def __init__(self):
        super(InterpolateBilinearFuser, self).__init__(graph_type="dygraph")

    def build_pattern(self):
        """ 描述需要替换的双线性插值图结构。
        interpolate_bilinear层模式python实现代码示例:
            x2834 = 'Exception'
            x2835 = None
            x2836 = 2
            x2837 = 3
            x2838 = 1
            x2839 = 0
            x2840 = 4
            x2841 = 5
            x2842 = None
            x2843 = x2832.shape
            x2843 = len(x2843)
            x2844 = x2843 - x2836
            x2845 = []
            for _x2847 in range(x2844):
                x2845.append(x2835)
            x2848 = (x2832, x9, x3, x3)
            x2849 = x2832.shape
            x2849 = len(x2849)
            x2850 = x2849 == x2837
            if x2850 :
                raise RaiseException(x2834)
                x2851 = x2842
            else:
                x2853 = x2832.shape
                x2853 = len(x2853)
                x2854 = x2853 == x2840
                if x2854 :
                    x2857 = True
                    x2858 = 'Exception'
                    x2859 = False
                    x2860 = None
                    x2861 = 'The default behavior for interpolate/upsample with float scale_factor will change in 1.6.0 to align with other frameworks/libraries, and use scale_factor directly, instead of relying on the computed output size. If you wish to keep the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details. '
                    x2862 = 0
                    x2863 = 9223372036854775807
                    x2864 = 1
                    x2865 = 2
                    x2866 = None
                    x2867 = None
                    x2868 = None
                    x2869 = None
                    x2870 = None
                    x2871, x2872, x2873, x2874 = x2848
                    x2875 = x2872 is x2860
                    if x2875 :
                        x2878 = x2873 is x2860
                        x2876 = x2878
                        x2877 = x2872
                    else:
                        x2879 = x2872
                        x2876 = x2859
                        x2877 = x2879
                    if x2876 :
                        raise RaiseException(x2858)
                    x2882 = x2877 is not x2860
                    if x2882 :
                        x2885 = x2877
                        x2886 = x2873 is not x2860
                        x2883 = x2886
                        x2884 = x2885
                    else:
                        x2883 = x2859
                        x2884 = x2877
                    if x2883 :
                        raise RaiseException(x2858)
                        x2887 = x2868
                        x2888 = x2869
                    else:
                        x2887 = x2873
                        x2888 = x2884
                    x2890 = x2887 is not x2860
                    if x2890 :
                        x2892 = x2887
                        x2893 = len(x2892)
                        x2894 = x2893 != x2836
                        if x2894 :
                            raise RaiseException(x2858)
                        x2891 = x2892
                    else:
                        x2891 = x2887
                    x2897 = x2888 is not x2860
                    if x2897 :
                        x2899 = x2888
                        x2898 = x2899
                    else:
                        x2898 = x2866
                    if x2897 :
                        x2900 = x2898
                    else:
                        x2901 = x2891 is not x2860
                        if x2901 :
                            x2903 = x2891
                            x2902 = x2903
                        else:
                            raise RaiseException(x2858)
                            x2902 = x2867
                        x2905 = x2874 is x2860
                        if x2905 :
                            x2907 = len(x2902)
                            x2908 = x2907 > x2862
                            x2912 = x2859
                            x2913 = x2862
                            for x2910 in range(x2863):
                                x2914 = x2902[x2913]
                                x2915 = math.floor(x2914)
                                x2916 = x2915 != x2914
                                if x2916 :
                                    x2917 = x2859
                                    x2918 = x2916
                                else:
                                    x2917 = x2870
                                    x2918 = x2870
                                if x2916 :
                                    x2919 = x2917
                                    x2920 = x2918
                                else:
                                    x2919 = x2857
                                    x2920 = x2916
                                x2921 = x2913 + x2864
                                x2922 = x2921 < x2907
                                x2923 = x2922 and x2919
                                x2909 = x2920
                                x2910 = x2921
                            if x2909 :
                                import warnings
                                warnings.warn(x2861, stacklevel=2)
                        x2926 = []
                        for _x2928 in range(x2836):
                            x2929 = _x2928 + x2865
                            x2930 = x2871.shape
                            x2931 = float(x2930)
                            x2932 = x2902[_x2928]
                            x2933 = x2931 * x2932
                            x2934 = math.floor(x2933)
                            x2926.append(x2934)
                        x2900 = x2926
                    x2935 = x2845[x2839]
                    x2936 = x2845[x2838]
                    assert x2935 == x2936, 'The x2935 must be x2936!'
                    x2937 = fluid.layers.interpolate(
                        input=x2832, out_shape=x2900, scale=x2935, align_corners=False, align_mode=0)
                    x2855 = x2937
                else:
                    x2938 = x2832.shape
                    x2938 = len(x2938)
                    x2939 = x2938 == x2841
                    if x2939 :
                        raise RaiseException(x2834)
                    else:
                        raise RaiseException(x2834)
                    x2855 = x2842
                x2851 = x2855
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "prim.constant",
            inputs={},
            outputs=[gen_name(0)],
            value="Exception")
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(1)], value=None)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(2)], value=2)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(3)], value=3)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(4)], value=1)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(5)], value=0)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(6)], value=4)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(7)], value=5)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(8)], value=None)
        self.pattern.add_layer(
            "prim.shape",
            inputs={"input": "interpolate-input-0"},
            outputs=[gen_name(9)])
        self.pattern.add_layer(
            "prim.len", inputs={"input": gen_name(9)}, outputs=[gen_name(9)])
        self.pattern.add_layer(
            "prim.sub",
            inputs={"x": gen_name(9),
                    "y": gen_name(2)},
            outputs=[gen_name(10)])
        self.pattern.add_layer("prim.list", inputs={}, outputs=[gen_name(11)])
        self.pattern.add_layer(
            "prim.loop",
            inputs={"input": gen_name(10)},
            outputs=[gen_name(12.1), gen_name(12.2)])
        loop_layer = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block = PaddleGraph(loop_layer, graph_type="dygraph")
        pattern_block.add_layer(
            "prim.append",
            inputs={"list": gen_name(11),
                    "element": gen_name(1)},
            outputs=[])
        loop_layer.inputs["input-0"] = gen_name(11)
        loop_layer.inputs["input-1"] = gen_name(1)
        loop_layer.add_block(pattern_block)
        self.pattern.add_layer(
            "prim.tuple",
            inputs={
                "input0": "interpolate-input-0",
                "input1": "interpolate-input-1",
                "input2": "interpolate-input-2",
                "input3": "interpolate-input-2"
            },
            outputs=[gen_name(13)])
        self.pattern.add_layer(
            "prim.shape",
            inputs={"input": "interpolate-input-0"},
            outputs=[gen_name(14)])
        self.pattern.add_layer(
            "prim.len", inputs={"input": gen_name(14)}, outputs=[gen_name(14)])
        self.pattern.add_layer(
            "prim.eq",
            inputs={"x": gen_name(14),
                    "y": gen_name(3)},
            outputs=[gen_name(15)])
        self.pattern.add_layer(
            "prim.if", inputs={"input": gen_name(15)}, outputs=[gen_name(16)])
        if_layer1 = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block = PaddleGraph(if_layer1, graph_type="dygraph")
        pattern_block.add_layer(
            "prim.exception",
            inputs={"input": gen_name(0)},
            outputs=[gen_name(17)])
        pattern_block.add_layer(
            "prim.equal", inputs={"input": gen_name(8)},
            outputs=[gen_name(16)])
        if_layer1.inputs["input-0"] = gen_name(0)
        if_layer1.inputs["input-1"] = gen_name(8)
        if_layer1.add_block(pattern_block)
        pattern_block = PaddleGraph(if_layer1, graph_type="dygraph")
        pattern_block.add_layer(
            "prim.shape",
            inputs={"input": "interpolate-input-0"},
            outputs=[gen_name(18)])
        pattern_block.add_layer(
            "prim.len", inputs={"input": gen_name(18)}, outputs=[gen_name(18)])
        pattern_block.add_layer(
            "prim.eq",
            inputs={"x": gen_name(18),
                    "y": gen_name(6)},
            outputs=[gen_name(19)])
        pattern_block.add_layer(
            "prim.if", inputs={"input": gen_name(19)}, outputs=[gen_name(20)])
        if_layer2 = pattern_block.layers[list(pattern_block.layers.keys())[-1]]
        pattern_block_block = PaddleGraph(if_layer2, graph_type="dygraph")
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(21)], value=False)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(22)], value=True)
        pattern_block_block.add_layer(
            "prim.constant",
            inputs={},
            outputs=[gen_name(23)],
            value="Exception")
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(24)], value=False)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(25)], value=None)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(26)], value="")
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(26.1)], value=0)
        pattern_block_block.add_layer(
            "prim.constant",
            inputs={},
            outputs=[gen_name(26.2)],
            value=9223372036854775807)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(27)], value=1)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(28)], value=2)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(29)], value=None)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(30)], value=None)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(31)], value=None)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(32)], value=None)
        pattern_block_block.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(33)], value=None)
        pattern_block_block.add_layer(
            "prim.tuple_unpack",
            inputs={"input": gen_name(13)},
            outputs=[gen_name(34), gen_name(35), gen_name(36), gen_name(37)])
        pattern_block_block.add_layer(
            "prim.is",
            inputs={"x": gen_name(35),
                    "y": gen_name(25)},
            outputs=[gen_name(38)])
        pattern_block_block.add_layer(
            "prim.if",
            inputs={"input": gen_name(38)},
            outputs=[gen_name(39), gen_name(40)])
        if_layer3 = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(if_layer3, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.is",
            inputs={"x": gen_name(36),
                    "y": gen_name(25)},
            outputs=[gen_name(41)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(41)},
            outputs=[gen_name(39)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(35)},
            outputs=[gen_name(40)])
        if_layer3.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(if_layer3, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(35)},
            outputs=[gen_name(42)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(24)},
            outputs=[gen_name(39)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(35)},
            outputs=[gen_name(40)])
        if_layer3.add_block(pattern_block_block_block)
        if_layer3.inputs.update({
            "input-0": gen_name(36),
            'input-1': gen_name(25),
            'input-2': gen_name(35),
            'input-3': gen_name(35),
            'input-4': gen_name(24)
        })
        pattern_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(39)}, outputs=[gen_name(43)])
        if_layer4 = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(if_layer4, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.exception",
            inputs={"input": gen_name(23)},
            outputs=[gen_name(44)])
        if_layer4.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(if_layer4, graph_type="dygraph")
        if_layer4.add_block(pattern_block_block_block)
        if_layer4.inputs["input-0"] = gen_name(23)
        pattern_block_block.add_layer(
            "prim.isnot",
            inputs={"x": gen_name(40),
                    "y": gen_name(25)},
            outputs=[gen_name(45)])
        pattern_block_block.add_layer(
            "prim.if",
            inputs={"input": gen_name(45)},
            outputs=[gen_name(46), gen_name(47)])
        if_layer5 = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(if_layer5, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(40)},
            outputs=[gen_name(48)])
        pattern_block_block_block.add_layer(
            "prim.isnot",
            inputs={"x": gen_name(36),
                    "y": gen_name(25)},
            outputs=[gen_name(49)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(49)},
            outputs=[gen_name(46)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(48)},
            outputs=[gen_name(47)])
        if_layer5.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(if_layer5, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(24)},
            outputs=[gen_name(46)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(40)},
            outputs=[gen_name(47)])
        if_layer5.add_block(pattern_block_block_block)
        if_layer5.inputs.update({
            "input-0": gen_name(40),
            "input-1": gen_name(36),
            "input-2": gen_name(25),
            "input-3": gen_name(24),
            "input-4": gen_name(40)
        })
        pattern_block_block.add_layer(
            "prim.if",
            inputs={"input": gen_name(46)},
            outputs=[gen_name(50), gen_name(51)])
        if_layer6 = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(if_layer6, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.exception",
            inputs={"input": gen_name(23)},
            outputs=[gen_name(52)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(31)},
            outputs=[gen_name(50)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(32)},
            outputs=[gen_name(51)])
        if_layer6.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(if_layer6, graph_type="dygraph")
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
            "input-0": gen_name(23),
            "input-1": gen_name(31),
            "input-2": gen_name(32),
            "input-3": gen_name(36),
            "input-4": gen_name(47)
        })
        pattern_block_block.add_layer(
            "prim.isnot",
            inputs={"x": gen_name(50),
                    "y": gen_name(25)},
            outputs=[gen_name(53)])
        pattern_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(53)}, outputs=[gen_name(54)])
        if_layer7 = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(if_layer7, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(50)},
            outputs=[gen_name(55)])
        pattern_block_block_block.add_layer(
            "prim.len", inputs={"input": gen_name(55)}, outputs=[gen_name(56)])
        pattern_block_block_block.add_layer(
            "prim.ne",
            inputs={"x": gen_name(56),
                    "y": gen_name(2)},
            outputs=[gen_name(57)])
        pattern_block_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(57)}, outputs=[gen_name(58)])
        if_layer8 = pattern_block_block_block.layers[list(
            pattern_block_block_block.layers.keys())[-1]]
        pattern_block_block_block_block = PaddleGraph(
            if_layer8, graph_type="dygraph")
        pattern_block_block_block_block.add_layer(
            "prim.exception",
            inputs={"input": gen_name(23)},
            outputs=[gen_name(59)])
        if_layer8.add_block(pattern_block_block_block_block)
        pattern_block_block_block_block = PaddleGraph(
            if_layer8, graph_type="dygraph")
        if_layer8.add_block(pattern_block_block_block_block)
        if_layer8.inputs["input-0"] = gen_name(23)
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(55)},
            outputs=[gen_name(54)])
        if_layer7.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(if_layer7, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(50)},
            outputs=[gen_name(54)])
        if_layer7.add_block(pattern_block_block_block)
        if_layer7.inputs.update({
            "input-0": gen_name(50),
            "input-1": gen_name(2),
            "input-2": gen_name(23),
            "input-3": gen_name(50)
        })
        pattern_block_block.add_layer(
            "prim.isnot",
            inputs={"x": gen_name(51),
                    "y": gen_name(25)},
            outputs=[gen_name(60)])
        pattern_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(60)}, outputs=[gen_name(61)])
        if_layer9 = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(if_layer9, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(51)},
            outputs=[gen_name(62)])
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(62)},
            outputs=[gen_name(61)])
        if_layer9.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(if_layer9, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(29)},
            outputs=[gen_name(61)])
        if_layer9.add_block(pattern_block_block_block)
        if_layer9.inputs.update({
            "input-0": gen_name(51),
            "input-1": gen_name(29)
        })
        pattern_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(60)}, outputs=[gen_name(63)])
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
            inputs={"x": gen_name(54),
                    "y": gen_name(25)},
            outputs=[gen_name(64)])
        pattern_block_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(64)}, outputs=[gen_name(65)])
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
            inputs={"input": gen_name(23)},
            outputs=[gen_name(67)])
        pattern_block_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(30)},
            outputs=[gen_name(65)])
        if_layer11.add_block(pattern_block_block_block_block)
        if_layer11.inputs.update({
            "input-0": gen_name(54),
            "input-1": gen_name(23),
            "input-2": gen_name(30)
        })
        pattern_block_block_block.add_layer(
            "prim.is",
            inputs={"x": gen_name(37),
                    "y": gen_name(25)},
            outputs=[gen_name(68)])
        pattern_block_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(68)}, outputs=[gen_name(69)])
        if_layer12 = pattern_block_block_block.layers[list(
            pattern_block_block_block.layers.keys())[-1]]
        pattern_block_block_block_block = PaddleGraph(
            if_layer12, graph_type="dygraph")
        pattern_block_block_block_block.add_layer(
            "prim.len", inputs={"input": gen_name(65)}, outputs=[gen_name(70)])
        pattern_block_block_block_block.add_layer(
            "prim.gt",
            inputs={"x": gen_name(70),
                    "y": gen_name(26.1)},
            outputs=[gen_name(71)])
        pattern_block_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(24)},
            outputs=[gen_name(72)])
        pattern_block_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(26.1)},
            outputs=[gen_name(73)])
        pattern_block_block_block_block.add_layer(
            "prim.loop",
            inputs={"input": gen_name(26.2)},
            outputs=[gen_name(74), gen_name(75), gen_name(76)])
        loop_layer = pattern_block_block_block_block.layers[list(
            pattern_block_block_block_block.layers.keys())[-1]]
        pattern_loop_block = PaddleGraph(loop_layer, graph_type="dygraph")
        pattern_loop_block.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(65),
                    "element": gen_name(73)},
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
        pattern_loop_block_block = PaddleGraph(if_layer13, graph_type="dygraph")
        pattern_loop_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(24)},
            outputs=[gen_name(77)])
        pattern_loop_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(76)},
            outputs=[gen_name(78)])
        if_layer13.add_block(pattern_loop_block_block)
        pattern_loop_block_block = PaddleGraph(if_layer13, graph_type="dygraph")
        pattern_loop_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(33)},
            outputs=[gen_name(77)])
        pattern_loop_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(33)},
            outputs=[gen_name(78)])
        if_layer13.add_block(pattern_loop_block_block)
        if_layer13.inputs.update({
            "input-0": gen_name(24),
            "input-1": gen_name(76),
            "input-2": gen_name(33),
            "input-3": gen_name(33)
        })
        pattern_loop_block.add_layer(
            "prim.if",
            inputs={"input": gen_name(76)},
            outputs=[gen_name(79), gen_name(80)])
        if_layer14 = pattern_loop_block.layers[list(
            pattern_loop_block.layers.keys())[-1]]
        pattern_loop_block_block = PaddleGraph(if_layer14, graph_type="dygraph")
        pattern_loop_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(77)},
            outputs=[gen_name(79)])
        pattern_loop_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(78)},
            outputs=[gen_name(80)])
        if_layer14.add_block(pattern_loop_block_block)
        pattern_loop_block_block = PaddleGraph(if_layer14, graph_type="dygraph")
        pattern_loop_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(22)},
            outputs=[gen_name(79)])
        pattern_loop_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(76)},
            outputs=[gen_name(80)])
        if_layer14.add_block(pattern_loop_block_block)
        if_layer14.inputs.update({
            "input-0": gen_name(77),
            "input-1": gen_name(78),
            "input-2": gen_name(22),
            "input-3": gen_name(76)
        })
        pattern_loop_block.add_layer(
            "prim.add",
            inputs={"x": gen_name(73),
                    "y": gen_name(27)},
            outputs=[gen_name(81)])
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
            "input-1": gen_name(73),
            "input-2": gen_name(24),
            "input-3": gen_name(33),
            "input-4": gen_name(33),
            "input-5": gen_name(22),
            "input-6": gen_name(73),
            "input-7": gen_name(27),
            "input-8": gen_name(70)
        })
        pattern_block_block_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(74)}, outputs=[gen_name(84)])
        if_layer15 = pattern_block_block_block_block.layers[list(
            pattern_block_block_block_block.layers.keys())[-1]]
        pattern_block_block_block_block_block = PaddleGraph(
            if_layer15, graph_type="dygraph")
        pattern_block_block_block_block_block.add_layer(
            "prim.warnings",
            inputs={"input": gen_name(26)},
            outputs=[gen_name(85)],
            stacklevel=2)
        if_layer15.add_block(pattern_block_block_block_block_block)
        pattern_block_block_block_block_block = PaddleGraph(
            if_layer15, graph_type="dygraph")
        if_layer15.add_block(pattern_block_block_block_block_block)
        if_layer15.inputs["input-0"] = gen_name(26)
        if_layer12.add_block(pattern_block_block_block_block)
        pattern_block_block_block_block = PaddleGraph(
            if_layer12, graph_type="dygraph")
        if_layer12.add_block(pattern_block_block_block_block)
        if_layer12.inputs.update({
            "input-0": gen_name(65),
            "input-1": gen_name(26.1),
            "input-2": gen_name(26.2),
            "input-3": gen_name(65),
            "input-4": gen_name(24),
            "input-5": gen_name(33),
            "input-6": gen_name(33),
            "input-7": gen_name(22),
            "input-8": gen_name(27),
            "input-9": gen_name(26)
        })
        pattern_block_block_block.add_layer(
            "prim.list", inputs={}, outputs=[gen_name(86)])
        pattern_block_block_block.add_layer(
            "prim.loop",
            inputs={"input": gen_name(2)},
            outputs=[gen_name(87), gen_name(88)])
        loop_layer = pattern_block_block_block.layers[list(
            pattern_block_block_block.layers.keys())[-1]]
        pattern_loop_block = PaddleGraph(loop_layer, graph_type="dygraph")
        pattern_loop_block.add_layer(
            "prim.add",
            inputs={"x": gen_name(88),
                    "y": gen_name(28)},
            outputs=[gen_name(89)])
        pattern_loop_block.add_layer(
            "prim.shape",
            inputs={"input": gen_name(34)},
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
            "input-1": gen_name(28),
            "input-2": gen_name(34),
            "input-3": gen_name(65),
            "input-5": gen_name(86)
        })
        pattern_block_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(86)},
            outputs=[gen_name(63)])
        if_layer10.add_block(pattern_block_block_block)
        if_layer10.inputs.update({
            "input-0": gen_name(61),
            "input-1": gen_name(54),
            "input-2": gen_name(25),
            "input-3": gen_name(54),
            "input-4": gen_name(23),
            "input-5": gen_name(30),
            "input-6": gen_name(37),
            "input-7": gen_name(25),
            "input-8": gen_name(26.1),
            "input-9": gen_name(26.2),
            "input-10": gen_name(24),
            "input-11": gen_name(33),
            "input-12": gen_name(33),
            "input-13": gen_name(22),
            "input-14": gen_name(27),
            "input-15": gen_name(26),
            "input-16": gen_name(2),
            "input-17": gen_name(28),
            "input-18": gen_name(34)
        })
        pattern_block_block.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(11),
                    "element": gen_name(5)},
            outputs=[gen_name(95)])
        pattern_block_block.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(11),
                    "element": gen_name(4)},
            outputs=[gen_name(96)])
        pattern_block_block.add_layer(
            "prim.assert",
            inputs={"key": gen_name(95),
                    "value": gen_name(96)},
            outputs=[gen_name(97) + "_assert"],
            type="eq")
        pattern_block_block.add_layer(
            "fluid.layers.interpolate",
            inputs={
                "input": "interpolate-input-0",
                "out_shape": gen_name(63),
                "scale": gen_name(95)
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
            "prim.shape",
            inputs={"input": "interpolate-input-0"},
            outputs=[gen_name(98)])
        pattern_block_block.add_layer(
            "prim.len", inputs={"input": gen_name(98)}, outputs=[gen_name(98)])
        pattern_block_block.add_layer(
            "prim.eq",
            inputs={"x": gen_name(98),
                    "y": gen_name(7)},
            outputs=[gen_name(99)])
        pattern_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(99)}, outputs=[gen_name(100)])
        if_layer16 = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(
            if_layer16, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.exception",
            inputs={"input": gen_name(0)},
            outputs=[gen_name(101)])
        if_layer16.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(
            if_layer16, graph_type="dygraph")
        pattern_block_block_block.add_layer(
            "prim.exception",
            inputs={"input": gen_name(0)},
            outputs=[gen_name(102)])
        if_layer16.add_block(pattern_block_block_block)
        if_layer16.inputs.update({
            "input-0": gen_name(0),
            "input-1": gen_name(0)
        })
        pattern_block_block.add_layer(
            "prim.equal", inputs={"input": gen_name(8)},
            outputs=[gen_name(20)])
        if_layer2.add_block(pattern_block_block)
        if_layer2.inputs.update({
            "input-0": gen_name(13),
            "input-1": gen_name(2),
            "input-2": gen_name(2),
            "input-3": gen_name(11),
            "input-4": gen_name(5),
            "input-5": gen_name(11),
            "input-6": gen_name(4),
            "input-7": "interpolate-input-0",
            "input-8": "interpolate-input-0",
            "input-9": gen_name(7),
            "input-10": gen_name(0),
            "input-11": gen_name(0),
            "input-12": gen_name(8)
        })
        pattern_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(20)},
            outputs=[gen_name(16)])
        if_layer1.add_block(pattern_block)
        if_layer1.inputs.update({
            "input-0": gen_name(0),
            "input-1": gen_name(8),
            "input-2": "interpolate-input-0",
            "input-3": gen_name(6),
            "input-4": gen_name(13),
            "input-5": gen_name(2),
            "input-6": gen_name(2),
            "input-7": gen_name(11),
            "input-8": gen_name(5),
            "input-9": gen_name(11),
            "input-10": gen_name(4),
            "input-11": "interpolate-input-0",
            "input-12": "interpolate-input-0",
            "input-13": gen_name(7),
            "input-14": gen_name(0),
            "input-15": gen_name(0),
            "input-16": gen_name(8)
        })
        self.pattern.build(inputs={
            "input-0": "interpolate-input-0",
            "input-1": "interpolate-input-1",
            "input-2": "interpolate-input-2",
        })

    def insert_new_layer(self, graph, parameters, matches):
        new_layer = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layer
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers_id = list(matches.keys())
        layer = matches[layers_id[15]]
        out_shape = layer.inputs["input1"]
        layer = matches[layers_id[21]]
        outputs = layer.outputs
        layer = matches[layers_id[128]]
        layer.inputs.pop("scale")
        layer.inputs["out_shape"] = out_shape
        layer.outputs = outputs
        return layer
