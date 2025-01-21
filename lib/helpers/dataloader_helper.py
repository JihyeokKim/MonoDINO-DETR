import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

from torch.utils.data import DistributedSampler as _DistributedSampler
from lib.helpers import utils_helper

from torch.utils.data.sampler import Sampler
from typing import Optional
import itertools

from lib.helpers import comm
from . import samplers
from .collate_batch import BatchCollator
class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=4, batch_size=2, dist=False, training=True):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader

    if dist:
        if training:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        else:
            rank, world_size = utils_helper.get_dist_info()
            train_sampler = DistributedSampler(train_set, world_size, rank, shuffle=False)
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=(train_sampler is None) and training,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler)
    

    return train_set, train_loader, train_sampler

def build_testloader(cfg, workers=4, batch_size=2, dist=False, training=False):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader

    if dist:
        if training:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        else:
            rank, world_size = utils_helper.get_dist_info()
            test_sampler = DistributedSampler(test_set, world_size, rank, shuffle=False)
    else:
        test_sampler = None

    test_loader = DataLoader(dataset=test_set,
                              batch_size=batch_size,
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=(test_sampler is None) and training,
                              pin_memory=False,
                              drop_last=False,
                              sampler=test_sampler)
    

    return test_set, test_loader, test_sampler


def make_data_loader(cfg, workers=4, batch_size=2, dist=True, training=True):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    size = len(train_set)
    sampler = samplers.TrainingSampler(size)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )
    
    # collator = BatchCollator(0)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=workers,
        batch_sampler=batch_sampler,
        # collate_fn=collator,
        pin_memory=True,
        worker_init_fn=my_worker_init_fn,
    )
    

    return train_set, train_loader, batch_sampler


def make_test_loader(cfg, workers=4, batch_size=2, dist=True, training=False):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    size = len(test_set)
    sampler = samplers.InferenceSampler(size)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False
    )
    # collator = BatchCollator(0)
    num_workers = workers
    test_loader = torch.utils.data.DataLoader(
        test_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        # collate_fn=collator,
    )

    return test_set, test_loader, batch_sampler


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)