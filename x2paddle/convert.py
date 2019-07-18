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
from x2paddle.parser.tf_parser import TFParser
from x2paddle.optimizer.tf_optimizer import TFGraphOptimizer
from x2paddle.emitter.tf_emitter import TFEmitter

parser = TFParser('/ssd2/Jason/github/X2Paddle/tool/vgg16_None.pb',
                  in_nodes=['inputs'],
                  out_nodes=['output_boxes'],
                  in_shapes=[[-1, 416, 416, 3]])

optimizer = TFGraphOptimizer()
#parser.tf_graph.print()

emitter = TFEmitter(parser)
emitter.run()
