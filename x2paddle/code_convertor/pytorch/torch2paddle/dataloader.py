import copy
import traceback
import six
import sys
import multiprocessing as mp

from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler




class BatchCompose(object):
    def __init__(self):
        self.output_fields = mp.Manager().list([])
        self.lock = mp.Lock()

    def __call__(self, data):
        # parse output fields by first sample
        # **this shoule be fixed if paddle.io.DataLoader support**
        # For paddle.io.DataLoader not support dict currently,
        # we need to parse the key from the first sample,
        # BatchCompose.__call__ will be called in each worker
        # process, so lock is need here.
        if len(self.output_fields) == 0:
            self.lock.acquire()
            if len(self.output_fields) == 0:
                for k, v in data[0].items():
                    self.output_fields.append(k)
            self.lock.release()
            
        data = [[data[i][k].numpy() for k in self.output_fields]
                for i in range(len(data))]
        
        data = list(zip(*data))
        batch_data = [np.stack(d, axis=0) for d in data]
        return batch_data


class BaseDataLoader(object):

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
        # batch transfrom 
        self._batch_transforms = BatchCompose()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # batch sampler
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
        else:
            self._batch_sampler = batch_sampler

        try:
            self.dataloader = DataLoader(
                dataset=dataset,
                batch_sampler=self._batch_sampler,
                num_workers=num_workers,
                return_list=True,
                use_buffer_reader=True,
                use_shared_memory=False,
                timeout=timeout,
                worker_init_fn=worker_init_fn)
        except Exception:
            self.dataloader = DataLoader(
                dataset=dataset,
                batch_sampler=self._batch_sampler,
                collate_fn=self._batch_transforms,
                num_workers=num_workers,
                return_list=True,
                use_buffer_reader=True,
                use_shared_memory=False,
                timeout=timeout,
                worker_init_fn=worker_init_fn)
        self.loader = iter(self.dataloader)

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        # pack {filed_name: field_data} here
        # looking forward to support dictionary
        # data structure in paddle.io.DataLoader
        try:
            data = next(self.loader)
            try:
                return data
            except Exception:
                return {
                    k: v
                    for k, v in zip(self._batch_transforms.output_fields, data)
                }
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())

    def next(self):
        # python2 compatibility
        return self.__next__()