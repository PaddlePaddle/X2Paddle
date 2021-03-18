import paddle
from paddle.nn import Sequential

Sequential.tmp = Sequential.__getitem__

def __getitem__(self, name):
    if isinstance(name, slice):
        return self.__class__(*(list(self._sub_layers.values())[name]))
    else:
        if name >= len(self._sub_layers):
            raise IndexError('index {} is out of range'.format(name))
        elif name < 0:
            while name < 0:
                name += len(self._sub_layers)
        return self.tmp(name)
    

Sequential.__getitem__ = __getitem__