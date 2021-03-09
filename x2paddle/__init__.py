__version__ = "1.0.1"

from .core.program import PaddleGraph
from .code_convertor.pytorch import torch2paddle

program = PaddleGraph()
