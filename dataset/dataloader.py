import torch 
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,


)
import time 
from torch.utils.data.dataloader import default_collate

class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False, epoch: int = 0):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def identity(x):
    return x


def create_mixed_dataloaders(
    dataset, batch_size, num_workers, world_size=None, rank=None, epoch=0, 
    image_mix_ratio=0.1, use_image_video_mixed_training=True,
):
    
    
    

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_collate,
        drop_last=True,
    )

    # To make it infinite
    loader = IterLoader(loader, use_distributed=True, epoch=epoch)
    return loader