from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


class ValidationWrapper(Dataset):
    """Wraps a dataset so that PyTorch Lightning's validation step can be turned into a
    visualization step.
    """

    dataset: Dataset
    dataset_samples: list
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        
        # Convert IterableDataset to list for random sampling
        if isinstance(dataset, IterableDataset):
            print("Converting IterableDataset to list for validation sampling...")
            self.dataset_samples = list(dataset)
            print(f"Validation dataset has {len(self.dataset_samples)} samples")
        else:
            self.dataset_samples = None

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if self.dataset_samples is not None:
            # Sample randomly from the converted list
            random_index = torch.randint(0, len(self.dataset_samples), tuple())
            return self.dataset_samples[random_index.item()]
        else:
            # For regular datasets, sample randomly
            random_index = torch.randint(0, len(self.dataset), tuple())
            return self.dataset[random_index.item()]
