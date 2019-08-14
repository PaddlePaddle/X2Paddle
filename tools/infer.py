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

import os
import sys
import inspect
import numpy as np


def load_data(imgfile, shape):
    h, w = shape[1:]
    from PIL import Image
    im = Image.open(imgfile)
    # The storage order of the loaded image is W(widht),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them.
    im = im.resize((w, h), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  # CHW
    im = im[(2, 1, 0), :, :]  # BGR
    # The mean to be subtracted from each image.
    # By default, the per-channel ImageNet mean.
    mean = np.array([104., 117., 124.], dtype=np.float32)
    mean = mean.reshape([3, 1, 1])
    im = im - mean
    im = im / 255.0
    return im.reshape([1] + list(im.shape))


def dump_results(results, names, root):
    if not os.path.exists(root):
        os.mkdir(root)
    for i in range(len(names)):
        n = names[i]
        res = np.array(results[i])
        filename = os.path.join(root, n)
        np.save(filename + '.npy', res)


def caffe_infer(prototxt_file, weight_file, datafile, save_path):
    """ do inference using pycaffe for debug,
        all intermediate results will be dumpped to 'results.caffe'
    """
    import caffe
    net = caffe.Net(prototxt_file, weight_file, caffe.TEST)
    input_layer = list(net.blobs.keys())[0]
    print('got name of input layer is:%s' % (input_layer))
    input_shape = list(net.blobs[input_layer].data.shape[1:])
    if '.npy' in datafile:
        np_images = np.load(datafile)
    else:
        np_images = load_data(datafile, input_shape)
    inputs = {input_layer: np_images}
    net.forward_all(**inputs)
    results = []
    names = []
    top_layer_dict = {}
    i = -1
    for layer_name, top_name in net.top_names.items():
        i += 1
        if net.layers[i].type == 'Dropout':
            continue
        for t in top_name:
            top_layer_dict[t] = layer_name
    for k, v in net.blobs.items():
        layer_name = top_layer_dict[k]
        layer_name = layer_name.replace('/', '_').replace('-', '_')
        if layer_name in names:
            index = names.index(layer_name)
            data = results[index]
            if isinstance(data, list):
                data.append(v.data[0].copy())
            else:
                data = [data]
                data.append(v.data[0].copy())
            results[index] = data
        else:
            names.append(layer_name)
            results.append(v.data[0].copy())

    dump_path = save_path + '/results.caffe'
    dump_results(results, names, dump_path)
    print('all caffe result of layers dumped to [%s]' % (dump_path))
    return 0


def paddle_infer(params_path, datafile, save_path):
    sys.path.append(params_path)
    import paddle.fluid as fluid
    from model import X2Paddle
    import model
    x2paddle_obj = X2Paddle()
    inputs, outputs = x2paddle_obj.x2paddle_net()
    input_names = [input.name for input in inputs]
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        b = os.path.exists(os.path.join(os.path.join(params_path, var.name)))
        return b

    fluid.io.load_vars(exe,
                       params_path,
                       fluid.default_main_program(),
                       predicate=if_exist)
    attrs = dir(x2paddle_obj)
    fetch_list = []
    names = []
    for attr in attrs:
        if '__' in attr or attr == 'x2paddle_net':
            continue
        else:
            attr_obj = getattr(x2paddle_obj, attr)
            if isinstance(attr_obj, list):
                for i, ao in enumerate(attr_obj):
                    fetch_list.append(ao)
                    names.append(attr + '-' + str(i))
            else:
                fetch_list.append(getattr(x2paddle_obj, attr))
                names.append(attr)
    input_shape = list(inputs[0].shape)[1:]
    if '.npy' in datafile:
        np_images = np.load(datafile)
    else:
        np_images = load_data(datafile, input_shape)
    results = exe.run(fluid.default_main_program(),
                      feed={'data': np_images},
                      fetch_list=fetch_list)
    dump_path = save_path + 'results.paddle'
    new_names = []
    new_results = []
    for i, name in enumerate(names):
        if '-' in name:
            part = name.split('-')
            if part[0] in new_names:
                index = new_names.index(part[0])
                data = new_results[index]
                if isinstance(data, list):
                    data.append(results[i])
                else:
                    data = [data]
                    data.append(results[i])
                new_results[index] = data
            else:
                new_names.append(part[0])
                new_results.append(results[i])
        else:
            new_names.append(name)
            new_results.append(results[i])
    dump_results(new_results, new_names, dump_path)
    print('all paddle result of layers dumped to [%s]' % (dump_path))
    return 0


if __name__ == "__main__":
    """ maybe more convenient to use 'run.sh' to call this tool
    """

    if len(sys.argv) <= 2:
        pass
    elif sys.argv[1] == 'caffe':
        if len(sys.argv) != 6:
            print('usage:')
            print(
                '\tpython %s caffe [prototxt] [caffemodel] [datafile] [save_path]'
                % (sys.argv[0]))
            sys.exit(1)
        prototxt_file = sys.argv[2]
        weight_file = sys.argv[3]
        datafile = sys.argv[4]
        save_path = sys.argv[5]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ret = caffe_infer(prototxt_file, weight_file, datafile, save_path)
    elif sys.argv[1] == 'paddle':
        if len(sys.argv) != 5:
            print('usage:')
            print('\tpython %s paddle [params_path] [datafile] [save_path]' %
                  (sys.argv[0]))
            sys.exit(1)
        params_path = sys.argv[2]
        datafile = sys.argv[3]
        save_path = sys.argv[4]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ret = paddle_infer(params_path, datafile, save_path)
