import os
import sys
import numpy as np
import onnx
import json
import argparse
from six import text_type as _text_type


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",
                        "-s",
                        type=_text_type,
                        default=None,
                        help="define save_dir")
    return parser


def main():
    try:
        import onnxruntime as rt
        version = rt.__version__
        if version != '0.4.0':
            print("onnxruntime==0.4.0 is required")
            return
    except:
        print(
            "onnxruntime is not installed, use \"pip install onnxruntime==0.4.0\"."
        )
        return
    parser = arg_parser()
    args = parser.parse_args()

    save_dir = args.save_dir
    model_dir = os.path.join(save_dir, 'onnx_model_infer.onnx')

    model = onnx.load(model_dir)
    sess = rt.InferenceSession(model_dir)

    inputs_dict = {}
    for ipt in sess.get_inputs():
        data_dir = os.path.join(save_dir, ipt.name + '.npy')
        inputs_dict[ipt.name] = np.load(data_dir, allow_pickle=True)
    res = sess.run(None, input_feed=inputs_dict)
    for idx, value_info in enumerate(model.graph.output):
        np.save(os.path.join(save_dir, value_info.name), res[idx])


if __name__ == "__main__":
    main()
