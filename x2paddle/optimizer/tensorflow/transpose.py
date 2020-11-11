import copy
import sys


class TransposeOpt:
    def __init__(self):
        self.image_layers = [
            'fluid.layers.conv2d', 'fluid.layers.batch_norm',
            'fluid.layers.conv2d_transpose', 'fluid.layers.resize_nearest',
            'fluid.layers.resize_bilinear', 'fluid.layers.pool2d',
            'fluid.layers.pad2d'
        ]
        self.direct_layers = [
            'fluid.layers.relu', 'fluid.layers.relu6', 'fluid.layers.abs',
            'fluid.layers.sigmoid', 'fluid.layers.exp', 'fluid.layers.rsqrt',
            'fluid.layers.swish_f32', 'fluid.layers.tanh',
            'fluid.layers.softplus', 'fluid.layers.leaky_relu',
            'fluid.layers.floor', 'fluid.layers.erf', 'fluid.layers.swish'
        ]
        self.elementwise_layers = [
            'fluid.layers.elementwise_add', 'fluid.layers.elementwise_sub',
            'fluid.layers.elementwise_mul', 'fluid.layers.elementwise_div'
        ]
        #        self.reduce_layers = []
        self.reduce_layers = [
            'fluid.layers.reduce_mean', 'fluid.layers.reduce_all',
            'fluid.layers.reduce_max', 'fluid.layers.reduce_any',
            'fluid.layers.reduce_sum', 'fluid.layers.reduce_prod'
        ]

    def get_transpose_num(self, graph):
        count = 0
        for layer_id, layer in graph.layers.items():
            if layer.kernel == "fluid.layers.transpose":
                count += 1
        return count

    def run(self, graph):
        print("Optimize: TransposeOpt...")
        total_layer_num = len(graph.layers)
        scanned_layers = set()
        optimized_transpose_layers = list()
        optimized_reduce_layers = list()
        optimized_concat_layers = list()
        optimized_elementwise_layers = list()

        def strip_transpose(_graph):
            layers = copy.deepcopy(_graph.layers)
            for layer_id, layer in layers.items():
                if layer_id in scanned_layers:
                    continue
                scanned_layers.add(layer_id)
                percent = round(len(scanned_layers) / total_layer_num * 100, 2)
                sys.stderr.write("\rOptimize Transpose Layers...{}%".format(
                    percent))

                if layer.kernel != "fluid.layers.transpose":
                    continue
                if layer.attrs["perm"] != [0, 2, 3, 1]:
                    continue
                transpose_layers = list()
                propagate_layers = list()
                reduce_layers = list()
                concat_layers = list()
                # 此elementwise_layers专用于存储shape(4) + shape(1)的形式layer
                elementwise_layers = list()
                can_be_optimized = True
                for out in _graph.edges_out.get(layer_id, []):
                    if _graph.layers[out].kernel == "fluid.layers.transpose":
                        if _graph.layers[out].attrs["perm"] != [0, 3, 1, 2]:
                            can_be_optimized = False
                            break
                        transpose_layers.append(out)
                    elif _graph.layers[out].kernel in self.elementwise_layers:
                        propagate_layers.append(out)
                    elif _graph.layers[out].kernel in self.direct_layers:
                        if _graph.layers[out].outputs[0] in _graph.outputs:
                            can_be_optimized = False
                            break
                        propagate_layers.append(out)
                    elif _graph.layers[out].kernel in self.reduce_layers:
                        if _graph.layers[out].outputs[0] in _graph.outputs:
                            can_be_optimized = False
                            break
                        if not _graph.layers[out].attrs.get('keep_dim', False):
                            can_be_optimized = False
                            break
                        propagate_layers.append(out)
                        reduce_layers.append(out)
                    elif _graph.layers[out].kernel == "fluid.layers.concat":
                        if _graph.layers[out].outputs[0] in _graph.outputs:
                            can_be_optimized = False
                            break
                        propagate_layers.append(out)
                        concat_layers.append(out)
                    else:
                        can_be_optimized = False
                        break

                visited_layers = set()
                while len(propagate_layers) > 0 and can_be_optimized:
                    current_id = propagate_layers.pop(0)
                    visited_layers.add(current_id)
                    for out in _graph.edges_out.get(current_id, []):
                        if _graph.layers[
                                out].kernel == "fluid.layers.transpose":
                            if _graph.layers[out].attrs["perm"] != [0, 3, 1, 2]:
                                can_be_optimized = False
                                break
                            transpose_layers.append(out)
                        elif _graph.layers[
                                out].kernel in self.elementwise_layers:
                            if _graph.layers[out].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                propagate_layers.append(out)
                        elif _graph.layers[out].kernel in self.direct_layers:
                            if _graph.layers[out].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                propagate_layers.append(out)
                        elif _graph.layers[out].kernel in self.reduce_layers:
                            if _graph.layers[out].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if not _graph.layers[out].attrs.get('keep_dim',
                                                                False):
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                propagate_layers.append(out)
                                reduce_layers.append(out)
                        elif _graph.layers[out].kernel == "fluid.layers.concat":
                            if _graph.layers[out].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                propagate_layers.append(out)
                                concat_layers.append(out)
                        else:
                            can_be_optimized = False
                            break
                    for ipt in _graph.edges_in.get(current_id, []):
                        if _graph.layers[
                                current_id].kernel in self.elementwise_layers:
                            try:
                                x_shape = _graph.layers[
                                    current_id].input_shapes['x']
                                y_shape = _graph.layers[
                                    current_id].input_shapes['y']
                                if _graph.layers[ipt].outputs[
                                        0] == _graph.layers[current_id].inputs[
                                            'x']:
                                    if len(x_shape) <= 1:
                                        elementwise_layers.append(current_id)
                                        continue
                                elif _graph.layers[ipt].outputs[
                                        0] == _graph.layers[current_id].inputs[
                                            'y']:
                                    if len(y_shape) <= 1:
                                        elementwise_layers.append(current_id)
                                        continue
                                else:
                                    raise Exception(
                                        "Unexcepted situation happend while optimizing transpose"
                                    )
                            except Exception as e:
                                can_be_optimized = False
                                break
                        if _graph.layers[
                                ipt].kernel == "fluid.layers.transpose":
                            if _graph.layers[ipt].attrs["perm"] != [0, 2, 3, 1]:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                transpose_layers.append(ipt)
                        elif _graph.layers[
                                ipt].kernel in self.elementwise_layers:
                            if _graph.layers[ipt].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                propagate_layers.append(ipt)
                        elif _graph.layers[ipt].kernel in self.direct_layers:
                            if _graph.layers[ipt].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                propagate_layers.append(ipt)
                        elif _graph.layers[ipt].kernel in self.reduce_layers:
                            if _graph.layers[ipt].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if not _graph.layers[ipt].attrs.get('keep_dim',
                                                                False):
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                propagate_layers.append(ipt)
                                reduce_layers.append(ipt)
                        elif _graph.layers[ipt].kernel == "fluid.layers.concat":
                            if _graph.layers[ipt].outputs[0] in _graph.outputs:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                propagate_layers.append(ipt)
                                concat_layers.append(ipt)
                        else:
                            can_be_optimized = False
                            break
                    if not can_be_optimized:
                        break
                if not can_be_optimized:
                    continue

                transpose_layers.append(layer_id)
                transpose_layers = list(set(transpose_layers))
                for l in transpose_layers:
                    if graph.layers[l].outputs[0] in graph.outputs:
                        can_be_optimized = False
                        break
                if not can_be_optimized:
                    continue

                for l in transpose_layers:
                    _graph.del_layer(l)

                optimized_transpose_layers.extend(transpose_layers)
                optimized_reduce_layers.extend(reduce_layers)
                optimized_concat_layers.extend(concat_layers)
                optimized_elementwise_layers.extend(elementwise_layers)
                return True
            return False

        before_transpose_num = self.get_transpose_num(graph)
        opt_graph = copy.deepcopy(graph)
        total_layer_num = len(opt_graph.layers)

        while strip_transpose(opt_graph):
            pass

        for layer_id in list(set(optimized_transpose_layers)):
            graph.del_layer(layer_id)
        for layer_id in list(set(optimized_reduce_layers)):
            dim = graph.layers[layer_id].attrs.get('dim', None)
            if dim is not None:
                for i in range(len(dim)):
                    dim[i] = [0, 2, 3, 1][dim[i]]
                graph.layers[layer_id].attrs['dim'] = dim
        for layer_id in list(set(optimized_concat_layers)):
            axis = graph.layers[layer_id].attrs.get('axis', 0)
            graph.layers[layer_id].attrs['axis'] = [0, 2, 3, 1][axis]
        for layer_id in list(set(optimized_elementwise_layers)):
            axis = graph.layers[layer_id].attrs.get('axis', -1)
            graph.layers[layer_id].attrs['axis'] = [0, 2, 3, 1][axis]

        current_transpose_num = self.get_transpose_num(graph)
        print(
            "\nTranspose layers optimized, before: transpose_num={}, after: transpose_num={}".
            format(before_transpose_num, current_transpose_num))
