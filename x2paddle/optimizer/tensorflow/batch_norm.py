import copy
from collections import OrderedDict
from x2paddle.core.program import PaddleLayer


class BatchNormOpt:
    def __init__(self):
        pass

    def run(self, graph):
        print("Optimize: BatchNormOpt...")
        layers = copy.deepcopy(graph.layers)
        for layer_id, layer in layers.items():
            if layer.kernel != "fluid.layers.elementwise_add":
                continue
            axis = layer.attrs.get('axis', -1)
            if axis != -1 and axis != 3:
                continue

            input_ids0 = graph.edges_in[layer_id]
            mul_layer0 = graph.layers[input_ids0[0]]
            sub_layer0 = graph.layers[input_ids0[1]]
            
            if mul_layer0.kernel != "fluid.layers.elementwise_mul":
                continue
            if sub_layer0.kernel != "fluid.layers.elementwise_sub":
                continue
            
            axis = mul_layer0.attrs.get('axis', -1)
            if axis != -1 and axis != 3:
                continue
            axis = sub_layer0.attrs.get('axis', -1)
            if axis != -1 and axis != 0:
                continue
            if len(graph.edges_out.get(input_ids0[0], [])) != 1:
                continue
            if len(graph.edges_out.get(input_ids0[1], [])) != 1:
                continue

            input_ids1 = graph.edges_in[input_ids0[0]]
            nhwc_input = graph.layers[input_ids1[0]]
            mul_layer1 = graph.layers[input_ids1[1]]
            if mul_layer1.kernel != "fluid.layers.elementwise_mul":
                continue
            axis = mul_layer1.attrs.get('axis', -1)
            if axis != -1 and axis != 0:
                continue
            if len(graph.edges_out.get(input_ids1[1], [])) != 2:
                continue

            input_ids2 = graph.edges_in[input_ids0[1]]
            beta = graph.layers[input_ids2[0]]
            mul_layer2 = graph.layers[input_ids2[1]]
            if beta.kernel != "fluid.layers.create_parameter":
                continue
            axis = mul_layer2.attrs.get('axis', -1)
            if axis != -1 and axis != 0:
                continue
            if len(graph.edges_out.get(input_ids2[0], [])) != 1:
                continue
            if len(graph.edges_out.get(input_ids2[1], [])) != 1:
                continue
            if beta.outputs[0] not in graph.parameters:
                continue
            beta_shape = graph.parameters[beta.outputs[0]].shape
            if len(beta_shape) != 1:
                continue

            input_ids3 = graph.edges_in[input_ids2[1]]
            mean = graph.layers[input_ids3[0]]
            mul_layer3 = graph.layers[input_ids3[1]]
            if mean.kernel != "fluid.layers.create_parameter":
                continue
            axis = mul_layer3.attrs.get('axis', -1)
            if axis != -1 and axis != 0:
                continue
            if len(graph.edges_out.get(input_ids3[0], [])) != 1:
                continue
            if len(graph.edges_out.get(input_ids3[1], [])) != 2:
                continue
            if mul_layer3.id != mul_layer1.id:
                continue
            if mean.outputs[0] not in graph.parameters:
                continue
            mean_shape = graph.parameters[mean.outputs[0]].shape
            if mean_shape != beta_shape:
                continue

            input_ids4 = graph.edges_in[input_ids3[1]]
            rsqrt_layer = graph.layers[input_ids4[0]]
            gamma = graph.layers[input_ids4[1]]
            if rsqrt_layer.kernel != "fluid.layers.rsqrt":
                continue
            if gamma.kernel != "fluid.layers.create_parameter":
                continue
            if len(graph.edges_out.get(input_ids4[0], [])) != 1:
                continue
            if len(graph.edges_out.get(input_ids4[1], [])) != 1:
                continue
            if gamma.outputs[0] not in graph.parameters:
                continue
            gamma_shape = graph.parameters[gamma.outputs[0]].shape
            if gamma_shape != beta_shape:
                continue

            input_ids5 = graph.edges_in[input_ids4[0]]
            add_layer = graph.layers[input_ids5[0]]
            if add_layer.kernel != "fluid.layers.elementwise_add":
                continue
            axis = add_layer.attrs.get('axis', -1)
            if axis != -1 and axis != 0:
                continue
            if len(graph.edges_out.get(input_ids5[0], [])) != 1:
                continue

            input_ids6 = graph.edges_in[input_ids5[0]]
            variance = graph.layers[input_ids6[0]]
            other = graph.layers[input_ids6[1]]
            if variance.kernel != "fluid.layers.create_parameter":
                continue
            if other.kernel != "fluid.layers.fill_constant":
                continue
            if len(graph.edges_out.get(input_ids6[0], [])) != 1:
                continue
            if len(graph.edges_out.get(input_ids6[1], [])) != 1:
                continue
            if variance.outputs[0] not in graph.parameters:
                continue
            variance_shape = graph.parameters[variance.outputs[0]].shape
            if variance_shape != beta_shape:
                continue

            ids = set([
                layer_id, mul_layer0.id, sub_layer0.id, mul_layer1.id, beta.id,
                mul_layer2.id, mean.id, mul_layer2.id, rsqrt_layer.id, gamma.id,
                add_layer.id, variance.id, other.id
            ])

            for id in ids:
                del graph.layers[id]
                if id in graph.edges_in:
                    del graph.edges_in[id]
                if id in graph.edges_out:
                    del graph.edges_out[id]

            copy_layers = copy.deepcopy(graph.layers)
            graph.layers = OrderedDict()
            for k, v in copy_layers.items():
                if k != nhwc_input.id:
                    graph.layers[k] = v
                    continue
                graph.layers[k] = v
                transpose0 = PaddleLayer(
                    id='{}_1'.format(k),
                    kernel="fluid.layers.transpose",
                    inputs={"x": v.outputs[0]},
                    outputs=["transpose_for_bn"],
                    perm=[0, 3, 1, 2])
                bn = PaddleLayer(
                    id='{}_2'.format(k),
                    kernel="fluid.layers.batch_norm",
                    inputs={"input": "transpose_for_bn"},
                    outputs=layer.outputs,
                    epsilon=other.attrs["value"],
                    param_attr="'{}'".format(gamma.outputs[0]),
                    bias_attr="'{}'".format(beta.outputs[0]),
                    moving_mean_name="'{}'".format(mean.outputs[0]),
                    moving_variance_name="'{}'".format(variance.outputs[0]))
                transpose1 = PaddleLayer(
                    id=layer_id,
                    kernel="fluid.layers.transpose",
                    inputs={"x": layer.outputs[0]},
                    outputs=layer.outputs,
                    perm=[0, 2, 3, 1])
                graph.layers[transpose0.id] = transpose0
                graph.layers[bn.id] = bn
                graph.layers[transpose1.id] = transpose1
        graph.build()
