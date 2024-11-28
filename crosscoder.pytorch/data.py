import time
from multiprocessing import Process, shared_memory

import numpy as np
import torch
from tqdm import tqdm


class IntermediateStateDataset(torch.utils.data.Dataset):
    def __init__(self, path1, path2):

        self.layer1 = np.load(path1, mmap_mode="r")
        self.layer2 = np.load(path2, mmap_mode="r")

        # self.buffer_size = 1000
        # self.buffer_pointer = 0
        # # note: this can be loaded first with `/dev/shm`
        # self.buffer1 = self.layer1[: self.buffer_size].copy()
        # self.buffer2 = self.layer2[: self.buffer_size].copy()

    def __len__(self):
        return self.layer1.shape[0]

    def __getitem__(self, idx):
        # if idx >= self.buffer_size + self.buffer_pointer:
        #     self.buffer_pointer += self.buffer_size
        #     tqdm.write(f"Loading new buffer from {self.buffer_pointer}")
        #     # 1. load these chunks asynchronously
        #     # 2. allow it to be accessible across processes
        #     self.buffer1 = self.layer1[
        #         self.buffer_pointer : self.buffer_pointer + self.buffer_size
        #     ].copy()
        #     self.buffer2 = self.layer2[
        #         self.buffer_pointer : self.buffer_pointer + self.buffer_size
        #     ].copy()
        # elif idx < self.buffer_pointer:
        #     tqdm.write(f"Loading new buffer from 0")
        #     self.buffer_pointer = 0
        #     self.buffer1 = self.layer1[: self.buffer_size].copy()
        #     self.buffer2 = self.layer2[: self.buffer_size].copy()

        # local_idx = idx % self.buffer_size

        # item1 = self.buffer1[local_idx]
        # item2 = self.buffer2[local_idx]
        # item = np.stack([item1, item2], axis=1)

        # return item

        item1 = self.layer1[idx]
        item2 = self.layer2[idx]
        item = np.stack([item1, item2], axis=1)
        return item


def load_to_shared_memory(path1, name1, path2, name2, start, end):
    data = np.load(path1, mmap_mode="r")
    shm = shared_memory.SharedMemory(name=name1)
    array = np.ndarray((end - start, 1024, 768), dtype=np.float32, buffer=shm.buf)
    array[:] = data[start:end].copy()
    data = np.load(path2, mmap_mode="r")
    shm = shared_memory.SharedMemory(name=name2)
    array = np.ndarray((end - start, 1024, 768), dtype=np.float32, buffer=shm.buf)
    array[:] = data[start:end].copy()


class IntermediateStateDatasetv2(torch.utils.data.Dataset):
    """
    Assumption: maximum disk read is faster than processing.
    """
    def __init__(self, path1, path2, buffer_size=1000):
        self.path1 = path1
        self.path2 = path2

        self.layer1 = np.load(self.path1, mmap_mode="r")
        self.layer2 = np.load(self.path2, mmap_mode="r")

        self.buffer_size = buffer_size
        self.buffer_pointer = 0
        self.buffer_nbytes = self.layer1[:self.buffer_size].nbytes

        self.nbuffer1 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)
        self.nbuffer2 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)
        load_to_shared_memory(self.path1, self.nbuffer1.name, self.path2, self.nbuffer2.name, 0, self.buffer_size)

        self.buffer1 = np.ndarray((self.buffer_size, 1024, 768), dtype=np.float32, buffer=self.nbuffer1.buf)
        self.buffer2 = np.ndarray((self.buffer_size, 1024, 768), dtype=np.float32, buffer=self.nbuffer2.buf)

        self.nbuffer1.close()
        self.nbuffer2.close()
        self.nbuffer1 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)
        self.nbuffer2 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)

        self.loading_process = Process(
            target=load_to_shared_memory,
            args=(
                self.path1, self.nbuffer1.name,
                self.path2, self.nbuffer2.name,
                self.buffer_pointer + self.buffer_size, self.buffer_pointer + 2 * self.buffer_size
            )
        )
        self.loading_process.start()

    def __len__(self):
        return self.layer1.shape[0]

    def __getitem__(self, idx):
        if idx >= self.buffer_size + self.buffer_pointer:
            if self.loading_process is not None and self.loading_process.is_alive():
                start = time.time()
                self.loading_process.join()
                print(f"Joining took {time.time() - start}")
                self.loading_process = None

            self.buffer_pointer += self.buffer_size
            self.buffer1 = np.ndarray((self.buffer_size, 1024, 768), dtype=np.float32, buffer=self.nbuffer1.buf)
            self.buffer2 = np.ndarray((self.buffer_size, 1024, 768), dtype=np.float32, buffer=self.nbuffer2.buf)

            self.nbuffer1.close()
            self.nbuffer1.unlink()
            self.nbuffer2.close()
            self.nbuffer2.unlink()
            self.nbuffer1 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)
            self.nbuffer2 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)

            self.loading_process = Process(
                target=load_to_shared_memory,
                args=(
                    self.path1, self.nbuffer1.name,
                    self.path2, self.nbuffer2.name,
                    self.buffer_pointer + self.buffer_size, self.buffer_pointer + 2 * self.buffer_size
                )
            )
            self.loading_process.start()

        local_idx = idx % self.buffer_size
        item1 = self.buffer1[local_idx]
        item2 = self.buffer2[local_idx]
        item = np.stack([item1, item2], axis=1)
        return item
