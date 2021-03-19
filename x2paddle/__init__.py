__version__ = "1.0.1"

from .core.program import PaddleGraph

program = PaddleGraph()

from x2paddle.code_convertor.pytorch import torch2paddle