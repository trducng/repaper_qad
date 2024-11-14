import numpy as np
import torch
from tqdm import tqdm


class IntermediateStateDataset(torch.utils.data.Dataset):
    def __init__(self, path1, path2):

        self.layer1 = np.load(path1, mmap_mode="r")
        self.layer2 = np.load(path2, mmap_mode="r")

        self.buffer_size = 3000
        self.buffer_pointer = 0
        self.buffer1 = self.layer1[: self.buffer_size]
        self.buffer2 = self.layer2[: self.buffer_size]

    def __len__(self):
        return self.layer1.shape[0]

    def __getitem__(self, idx):
        if idx >= self.buffer_size + self.buffer_pointer:
            self.buffer_pointer += self.buffer_size
            tqdm.write(f"Loading new buffer from {self.buffer_pointer}")
            self.buffer1 = self.layer1[
                self.buffer_pointer : self.buffer_pointer + self.buffer_size
            ]
            self.buffer2 = self.layer2[
                self.buffer_pointer : self.buffer_pointer + self.buffer_size
            ]

        local_idx = idx % self.buffer_size

        item1 = self.layer1[local_idx]
        item2 = self.layer2[local_idx]
        item = np.stack([item1, item2], axis=0)
        item = item.transpose(1, 0, 2)

        return item
