import os
import json
from pathlib import Path
import mmap
import time
from multiprocessing import Process, shared_memory

import humanize
import numpy as np
import torch
from tqdm import tqdm

from dawnet import Inspector


def count_lines_file(fp):
    fp = Path(fp)

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        print(
            f"Number of lines for {humanize.intcomma(fp.name)}:",
            sum(bl.count("\n") for bl in blocks(f)),
        )


def count_lines(directory):
    for fp in Path(directory).glob("*.jsonl"):
        start = time.time()
        count_lines_file(fp)
        print(f"  Time taken: {time.time() - start}")


def tokenize_file_to_jsonl(input_path, output_path):
    start = time.time()
    from transformers import AutoTokenizer

    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    n_tokens = 0
    with open(output_path, "w") as fo:
        with open(input_path) as fi:
            for idx, line in enumerate(fi):
                if idx > 0:
                    fo.write("\n")
                if idx % 10000 == 0:
                    print(f" Line {idx}, {time.time() - start}")
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error at line {idx}")
                    continue
                text = data["text"]
                tokens = tokenizer.encode(text)
                n_tokens += len(tokens)
                fo.write(json.dumps(tokens).replace(" ", ""))
    print(f"Time taken: {time.time() - start}")
    print(f"Number of tokens: {n_tokens}")


def tokenize_directory_to_jsonl(directory, output_dir):
    for fp in sorted(Path(directory).glob("*.jsonl")):
        print(f"Tokenizing {fp}")
        tokenize_file_to_jsonl(fp, Path(output_dir) / fp.name)


def tokenize_file_to_bin(input_path, output_path, start_idx=0):
    start = time.time()
    from transformers import AutoTokenizer

    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    n_tokens = 0
    if start_idx > 0 and Path(output_path).exists():
        mode = "ab"
    else:
        mode = "wb"
    with open(output_path, mode) as fo:
        with open(input_path) as fi:
            for idx, line in enumerate(fi):
                if idx < start_idx:
                    continue
                if idx % 10000 == 0:
                    print(f" Line {idx}, {time.time() - start}")
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error at line {idx}")
                    continue
                text = data["text"]
                tokens = tokenizer.encode(text)
                n_tokens += len(tokens)
                for token in tokens:
                    fo.write(token.to_bytes(2, "big"))
                fo.write(tokenizer.eos_token_id.to_bytes(2, "big"))
    print(f"Time taken: {time.time() - start}")
    print(f"Number of tokens: {n_tokens}")


def tokenize_directory_to_bin(directory, output_dir):
    for fp in sorted(Path(directory).glob("*.jsonl")):
        print(f"Tokenizing {fp}")
        tokenize_file_to_bin(fp, (Path(output_dir) / fp.name).with_suffix(".bin"))
        break


def annotate_start_end_bytes(
    input_path,
    bytes_per_token=2,
    sep_token_id=50256,
    output_path=None,
    resume: bool = True,
):
    """Annotate start and end bytes for fast data access"""
    start = time.time()
    if resume and output_path is not None and Path(output_path).exists():
        with open(output_path, "r") as fi:
            position = json.load(fi)
            print(f"Resuming from {len(position)}")
            start_position = position[-1][1] + bytes_per_token
    else:
        position = []
        start_position = 0
    with open(input_path, "rb") as fi:
        fi.seek(start_position)
        token = fi.read(bytes_per_token)
        while token:
            value = int.from_bytes(token, "big")
            if value == sep_token_id:
                position.append((start_position, fi.tell() - bytes_per_token))
                start_position = fi.tell()
            token = fi.read(bytes_per_token)
    print(f"Time taken: {time.time() - start}")
    if output_path is not None:
        with open(output_path, "w") as fo:
            json.dump(position, fo)
    return position


def iter_lines_from_bin(input_path, sep_token_id=50256):
    with open(input_path, "rb") as fi:
        tokens = []
        token = fi.read(2)
        while token:
            value = int.from_bytes(token, "big")
            if value == sep_token_id:
                yield tokens
                tokens = []
            else:
                tokens.append(value)
            token = fi.read(2)


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
        self.buffer_nbytes = self.layer1[: self.buffer_size].nbytes

        self.nbuffer1 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)
        self.nbuffer2 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)
        load_to_shared_memory(
            self.path1,
            self.nbuffer1.name,
            self.path2,
            self.nbuffer2.name,
            0,
            self.buffer_size,
        )

        self.buffer1 = np.ndarray(
            (self.buffer_size, 1024, 768), dtype=np.float32, buffer=self.nbuffer1.buf
        )
        self.buffer2 = np.ndarray(
            (self.buffer_size, 1024, 768), dtype=np.float32, buffer=self.nbuffer2.buf
        )

        self.nbuffer1.close()
        self.nbuffer2.close()
        self.nbuffer1 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)
        self.nbuffer2 = shared_memory.SharedMemory(create=True, size=self.buffer_nbytes)

        self.loading_process = Process(
            target=load_to_shared_memory,
            args=(
                self.path1,
                self.nbuffer1.name,
                self.path2,
                self.nbuffer2.name,
                self.buffer_pointer + self.buffer_size,
                self.buffer_pointer + 2 * self.buffer_size,
            ),
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
            self.buffer1 = np.ndarray(
                (self.buffer_size, 1024, 768),
                dtype=np.float32,
                buffer=self.nbuffer1.buf,
            )
            self.buffer2 = np.ndarray(
                (self.buffer_size, 1024, 768),
                dtype=np.float32,
                buffer=self.nbuffer2.buf,
            )

            self.nbuffer1.close()
            self.nbuffer1.unlink()
            self.nbuffer2.close()
            self.nbuffer2.unlink()
            self.nbuffer1 = shared_memory.SharedMemory(
                create=True, size=self.buffer_nbytes
            )
            self.nbuffer2 = shared_memory.SharedMemory(
                create=True, size=self.buffer_nbytes
            )

            self.loading_process = Process(
                target=load_to_shared_memory,
                args=(
                    self.path1,
                    self.nbuffer1.name,
                    self.path2,
                    self.nbuffer2.name,
                    self.buffer_pointer + self.buffer_size,
                    self.buffer_pointer + 2 * self.buffer_size,
                ),
            )
            self.loading_process.start()

        local_idx = idx % self.buffer_size
        item1 = self.buffer1[local_idx]
        item2 = self.buffer2[local_idx]
        item = np.stack([item1, item2], axis=1)
        return item


class IntermediateStateFromTokens(torch.utils.data.Dataset):
    """Get intermediate state by running the models through the token and
    capture the intermediate state
    """

    def __init__(
        self,
        token_path: str,
        token_annot_path: str,
        # layers: str | list[str],
        # inspector: Inspector,
    ):
        with open(token_annot_path, "r") as fi:
            self.token_annot = json.load(fi)

        # register
        self.token_path = token_path
        self.handler = None

    def __len__(self):
        return len(self.token_annot)

    def __getitem__(self, idx) -> list[int]:
        if self.handler is None:
            self.handler = open(self.token_path, "rb")
        start, end = self.token_annot[idx]
        tokens = []
        self.handler.seek(start)
        token = self.handler.read(2)
        while self.handler.tell() <= end:
            value = int.from_bytes(token, "big")
            tokens.append(value)
            token = self.handler.read(2)
        return tokens

    def close(self):
        if self.handler is not None:
            self.handler.close()


class IntermediateStateMMAPFromTokens(torch.utils.data.Dataset):
    """Get intermediate state by running the models through the token and
    capture the intermediate state
    """

    def __init__(
        self,
        token_path: str,
        token_annot_path: str,
        # layers: str | list[str],
        # inspector: Inspector,
    ):
        with open(token_annot_path, "r") as fi:
            self.token_annot = json.load(fi)

        # register
        self.token_path = token_path
        self.handler = None
        self.mmap_handler = None

    def __len__(self):
        return len(self.token_annot)

    def __getitem__(self, idx) -> list[int]:
        if self.handler is None or self.mmap_handler is None:
            self.handler = open(self.token_path, "rb")
            self.mmap_handler = mmap.mmap(
                self.handler.fileno(), length=5000, access=mmap.ACCESS_READ
            )

        start, end = self.token_annot[idx]
        tokens = []
        self.mmap_handler.seek(start)
        token = self.mmap_handler.read(2)
        while self.mmap_handler.tell() <= end:
            value = int.from_bytes(token, "big")
            tokens.append(value)
            token = self.mmap_handler.read(2)
        return tokens

    def close(self):
        if self.mmap_handler is not None:
            self.mmap_handler.close()
        if self.handler is not None:
            self.handler.close()


def check_annot_validity(input_path, annot_path, bin_path):
    with open(annot_path, "r") as fi:
        position = json.load(fi)

    # check if start and end bytes are valid
    for idx, (_, end) in enumerate(position):
        if idx == len(position) - 1:
            if os.path.getsize(bin_path) != end + 2:
                print(f"Error at last index: {idx}")
            break
        if position[idx + 1][0] != end + 2:
            raise ValueError(f"Error at index: {idx}: {end} - {position[idx + 1][0]}")

    # check if we have the correct number of lines
    with open(input_path, "r") as fi:
        count = 0
        for _ in fi:
            count += 1
        print(f"{count=}, {len(position)=}")
        if count != len(position):
            raise ValueError(f"Error: {count} != {len(position)}")


def count_tokens(path):
    count_total = 0
    for fp in sorted(Path(path).glob("*.annot.json")):
        count_file = 0
        with open(fp, "r") as fi:
            position = json.load(fi)
        for start, end in position:
            count_file += (end - start) // 2 - 1
        print(f"{fp.name}: {humanize.intcomma(count_file)}")
        count_total += count_file
    print(f"Total: {humanize.intcomma(count_total)}")


if __name__ == "__main__":
    # tokenize_file_to_jsonl(
    #     "/data2/datasets/thepile/train/02.jsonl",
    #     "/data3/mech/thepile_gpt2_tokenized/train/02.jsonl",
    # )
    # tokenize_file_to_bin(
    #     "/data2/datasets/thepile/train/25.jsonl",
    #     "/data3/mech/thepile_gpt2_tokenized/train/25.bin",
    #     start_idx=4549453,
    # )
    # tokenize_directory(
    #     "/data2/datasets/thepile/train/",
    #     "/data3/mech/thepile_gpt2_tokenized/train/",
    # )
    # tokenize_directory_to_bin(
    #     "/data2/datasets/thepile/train/",
    #     "/data3/mech/thepile_gpt2_tokenized/train/",
    # )

    # directory = "/data2/datasets/thepile/train/"
    # output_dir = "/data3/mech/thepile_gpt2_tokenized/train/"
    # files = list(sorted(Path(directory).glob("*.jsonl")))
    # for fp in files[26:30]:
    #     print(f"Tokenizing {fp}")
    #     tokenize_file_to_bin(fp, (Path(output_dir) / fp.name).with_suffix(".bin"))

    # directory = "/data3/mech/thepile_gpt2_tokenized/train"
    # output_dir = "/data3/mech/thepile_gpt2_tokenized/train"
    # files = list(sorted(Path(directory).glob("*.bin")))
    # for fp in files[2:17]:
    #     print(f"Annotating {fp}")
    #     annotate_start_end_bytes(
    #         fp,
    #         output_path=Path(output_dir) / f"{fp.stem}.annot.json",
    #     )

    # dataset = IntermediateStateMMAPFromTokens(
    #     token_path="/data3/mech/thepile_gpt2_tokenized/train/00.bin",
    #     token_annot_path="/home/john/temp/annot.json"
    # )

    # check_annot_validity(
    #     input_path="/data2/datasets/thepile/train/27.jsonl",
    #     annot_path="/data3/mech/thepile_gpt2_tokenized/train/27.annot.json",
    #     bin_path="/data3/mech/thepile_gpt2_tokenized/train/27.bin",
    # )

    # annotate_start_end_bytes(
    #     "/data3/mech/thepile_gpt2_tokenized/train/29.bin",
    #     output_path="/data3/mech/thepile_gpt2_tokenized/train/29.annot.json",
    # )

    # annotate_start_end_bytes(
    #     "/data3/mech/thepile_gpt2_tokenized/train/28.bin",
    #     output_path="/data3/mech/thepile_gpt2_tokenized/train/28.annot.json",
    # )

    # tokenize_file_to_bin(
    #     "/data2/datasets/thepile/train/28.jsonl",
    #     "/data3/mech/thepile_gpt2_tokenized/train/28.bin",
    #     start_idx=4151730,
    # )

    count_tokens("/data3/mech/thepile_gpt2_tokenized/train")
