## torch.utils.data.random_split
### [torch.utils.data.random_split](https://pytorch.org/docs/stable/data.html?highlight=random_split#torch.utils.data.random_split)
```python
torch.utils.data.random_split(dataset, lengths, generator=<torch._C.Generator object>)
```

### 功能介绍
用于实现数据集划分，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。

```python
import paddle
from paddle.io import Dataset
def _accumulate(iterable, fn=lambda x, y: x + y):
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def random_split(dataset, lengths, generator=None):
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = paddle.randperm(sum(lengths))
    return [
        Subset(dataset, indices[offset - length: offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]
```
