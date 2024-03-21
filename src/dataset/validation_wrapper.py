from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


class ValidationWrapper(Dataset):
    """Wraps a dataset so that PyTorch Lightning's validation step can be turned into a
    visualization step.
    """

    dataset: Dataset
    dataset_iterator: Optional[Iterator]
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if isinstance(self.dataset, IterableDataset):
            if self.dataset_iterator is None:
                self.dataset_iterator = iter(self.dataset)
            return next(self.dataset_iterator)

        random_index = torch.randint(0, len(self.dataset), tuple())
        return self.dataset[random_index.item()]
