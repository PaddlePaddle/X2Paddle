from paddle.io import Dataset, IterableDataset
from paddle.io import DataLoader as Base_DataLoader
import paddle
import warnings
import bisect

class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

def _accumulate(iterable, fn=lambda x, y: x + y):
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
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
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    

def random_split(dataset, lengths, generator=None):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): from torch import default_generator, which is not use in paddle.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = paddle.randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
setattr(paddle.io, "random_split", random_split)



class DataLoader(Base_DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1, 
                 shuffle=False, 
                 sampler=None, 
                 batch_sampler=None, 
                 num_workers=0, 
                 collate_fn=None, 
                 pin_memory=False, 
                 drop_last=False, 
                 timeout=0,
                 worker_init_fn=None, 
                 multiprocessing_context=None, 
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False
        return_list = True
        super().__init__(dataset,
                         feed_list=None, 
                         places=None, 
                         return_list=return_list, 
                         batch_sampler=batch_sampler, 
                         batch_size=batch_size, 
                         shuffle=shuffle, 
                         drop_last=drop_last, 
                         collate_fn=collate_fn, 
                         num_workers=num_workers, 
                         use_buffer_reader=True, 
                         use_shared_memory=False, 
                         timeout=timeout, 
                         worker_init_fn=worker_init_fn)
        if sampler is not None:
            seld.batch_sampler.sampler = sampler
