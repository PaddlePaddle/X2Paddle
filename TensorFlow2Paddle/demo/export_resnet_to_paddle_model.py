import sys
sys.path.append(".")
from transformer import Transformer

meta_file = sys.argv[1]
ckpt_dir = sys.argv[2]
export_dir = sys.argv[3]

transformer = Transformer(meta_file, ckpt_dir, ['resnet_v1_50/pool5'],
                          (224, 224, 3), ['inputs'])
transformer.run(export_dir)

open(export_dir + "/__init__.py", "w").close()
