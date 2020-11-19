import copy
import numpy as np
from collections import OrderedDict
from x2paddle.core.program import PaddleLayer
from x2paddle.core.util import *


class PReLUOpt:
    def __init__(self):
        pass

    def run(self, graph):
        print("Optimize: PReLUOpt...")
        layers = copy.deepcopy(graph.layers)
        for layer_id, layer in layers.items():
            if layer.kernel != "fluid.layers.elementwise_add":
                continue
            axis = layer.attrs.get('axis', -1)
            if axis != -1 and axis != 3:
                continue

            input_ids0 = graph.edges_in[layer_id]
            relu_layer0 = graph.layers[input_ids0[0]]
            mul_layer0 = graph.layers[input_ids0[1]]
            
            if relu_layer0.kernel != "fluid.layers.relu":
                continue
            if mul_layer0.kernel != "fluid.layers.elementwise_mul":
                continue
            
            axis = mul_layer0.attrs.get('axis', -1)
            if axis != -1 and axis != 3:
                continue
            if len(graph.edges_out.get(input_ids0[0], [])) != 1:
                continue
            if len(graph.edges_out.get(input_ids0[1], [])) != 1:
                continue
                
            input_ids1_0 = graph.edges_in[input_ids0[0]]
            input_ids1_1 = graph.edges_in[input_ids0[1]]
            fill_layer = graph.layers[input_ids1_1[1]]
            mul_layer1 = graph.layers[input_ids1_1[0]]
            if fill_layer.kernel != "fluid.layers.fill_constant":
                continue
            if mul_layer1.kernel != "fluid.layers.elementwise_mul":
                continue
            axis = mul_layer1.attrs.get('axis', -1)
            if axis != -1 and axis != 0:
                continue
            if len(graph.edges_out.get(input_ids1_1[1], [])) != 1:
                continue
            if len(graph.edges_out.get(input_ids1_0[0], [])) != 3:
                continue     
              
            input_ids2 = graph.edges_in[input_ids1_1[0]]    
            alpha = graph.layers[input_ids2[0]]
            sub_layer = graph.layers[input_ids2[1]]
            if alpha.kernel != "fluid.layers.create_parameter":
                continue
            if sub_layer.kernel != "fluid.layers.elementwise_sub":
                continue
            axis = sub_layer.attrs.get('axis', -1)
            if axis != -1 and axis != 3:
                continue
            if len(graph.edges_out.get(input_ids2[0], [])) != 1:
                continue
            if len(graph.edges_out.get(input_ids2[1], [])) != 1:
                continue
            if alpha.outputs[0] not in graph.parameters:
                continue
            
            input_ids3 = graph.edges_in[input_ids2[1]]
            add_layer = graph.layers[input_ids3[0]]
            abs_layer = graph.layers[input_ids3[1]]
            if abs_layer.kernel != "fluid.layers.abs":
                continue
            if len(graph.edges_out.get(input_ids3[1], [])) != 1:
                continue
                

            ids = set([
                layer.id, relu_layer0.id, mul_layer0.id, fill_layer.id, mul_layer1.id, alpha.id,
                sub_layer.id, abs_layer.id])

            for id in ids:
                del graph.layers[id]
                if id in graph.edges_in:
                    del graph.edges_in[id]
                if id in graph.edges_out:
                    del graph.edges_out[id]

            copy_layers = copy.deepcopy(graph.layers)
            graph.layers = OrderedDict()
            for k, v in copy_layers.items():
                if k != add_layer.id:
                    graph.layers[k] = v
                    continue
                graph.layers[k] = v
                transpose0 = PaddleLayer(
                    id='{}_1'.format(k),
                    kernel="fluid.layers.transpose",
                    inputs={"x": v.outputs[0]},
                    outputs=["transpose_for_prelu"],
                    perm=[0, 3, 1, 2])
                prelu = PaddleLayer(
                    id='{}_2'.format(k),
                    kernel="fluid.layers.prelu",
                    inputs={"x": "transpose_for_prelu"},
                    outputs=layer.outputs,
                    mode=string("channel"),
                    param_attr="'{}'".format(alpha.outputs[0]))
                transpose1 = PaddleLayer(
                    id=layer_id,
                    kernel="fluid.layers.transpose",
                    inputs={"x": layer.outputs[0]},
                    outputs=layer.outputs,
                    perm=[0, 2, 3, 1])
                graph.layers[transpose0.id] = transpose0
                graph.layers[prelu.id] = prelu
                graph.layers[transpose1.id] = transpose1
                graph.parameters[alpha.outputs[0]] = np.expand_dims(graph.parameters[alpha.outputs[0]], 0)
        graph.build()
        
