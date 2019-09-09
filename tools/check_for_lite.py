from six.moves import urllib
import sys
from paddle.fluid.framework import Program

ops_h = "https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite/develop/lite/api/_paddle_use_ops.h"
try:
    fetch = urllib.urlretrieve(ops_h, "./_paddle_use_ops.h")
except:
    fetch = urllib.request.urlretrieve(ops_h, "./_paddle_use_ops.h")

ops = list()
with open("./_paddle_use_ops.h") as f:
    for line in f:
        if "USE_LITE_OP" in line:
            op = line.strip().split('(')[1].split(')')[0]
            ops.append(op)

model_file = sys.argv[1]
with open(model_file, 'rb') as f:
    program = Program.parse_from_string(f.read())

unsupported_ops = set()
for op in program.blocks[0].ops:
    if op.type not in ops:
        unsupported_ops.add(op.type)

nums = len(unsupported_ops)
if len(unsupported_ops) > 0:
    print("========= {} OPs are not supported in Paddle-Lite=========".format(
        nums))
    for op in unsupported_ops:
        print("========= {} ========".format(op))
else:
    print("\n========== Good News! ========")
    print("Good! All ops in this model are supported in Paddle-Lite!\n")
