from x2paddle.decoder.caffe_decoder import CaffeDecoder
Decoder = CaffeDecoder

from x2paddle.op_mapper.dygraph.caffe2paddle.caffe_op_mapper import CaffeOpMapper
DygraphOpMapper = CaffeOpMapper

from x2paddle.op_mapper.static.caffe2paddle.caffe_op_mapper import CaffeOpMapper
StaticOpMapper = CaffeOpMapper

from x2paddle.optimizer.caffe_optimizer import CaffeOptimizer
StaticOptimizer = CaffeOptimizer