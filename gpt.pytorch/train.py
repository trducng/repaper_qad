import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gpt1 import Transformer
from hug_tokenizers import get_bookcorpus_tokenizer


class BookDataset(Dataset):
    def __init__(self):
        self.tokenizer = get_bookcorpus_tokenizer("downloads/tokenizer_bookcorpus.json")
        self.data = load_dataset("bookcorpus")

    def __len__(self):
        return len(self.data["train"])

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.data["train"][idx]["text"].lower()).ids[:512]
        while len(x) <= 1:
            new_idx = random.randrange(len(self.data["train"]))
            x = self.tokenizer.encode(self.data["train"][new_idx]["text"].lower()).ids[:512]
        x += [self.tokenizer.token_to_id("[PAD]")] * (512 - len(x))
        return x[:-1], x[1:]


dataset = BookDataset()


def collate(items):
    x, y = zip(*items)
    max_len = max([len(each) for each in x])
    new_x = torch.LongTensor([
        each + [dataset.tokenizer.token_to_id("[PAD]")] * (max_len - len(each))
        for each in x]
    )
    new_y = torch.LongTensor(
        [each + [dataset.tokenizer.token_to_id("[PAD]")] * (max_len - len(each))
        for each in y]
    )
    return new_x, new_y


dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate
)


model = Transformer(
    vocab_size=dataset.tokenizer.get_vocab_size(),
    embedding_dim=192,
    sequence_length=512,
    n_blocks=6,
    n_heads=6,
)
model.train()
model = model.cuda()

decay_params, non_decay_params = [], []
for key, value in model.named_parameters():
    if "embedding" in key or "norm" in key or "bias" in key:
        non_decay_params.append(value)
    else:
        decay_params.append(value)
model_params = [
    {"params": decay_params, "weight_decay": 0.01},
    {"params": non_decay_params, "weight_decay": 0.0},
]
optimizer = optim.Adam(model_params, lr=2.5e-4 / 2000)
epochs = 100
lr_scheduler = optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda epoch: epoch,
    last_epoch=-1
)


for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    pbar = tqdm(dataloader, total=len(dataloader))
    for idx, (x, y) in enumerate(pbar):
        model.zero_grad(set_to_none=True)
        x, y = x.cuda(), y.cuda()
        try:
            pred = model(x)
            loss = F.cross_entropy(pred.view(-1, pred.shape[-1]), y.flatten())
            loss.backward()
            optimizer.step()
            if epoch == 0 and idx < 2000:
                lr_scheduler.step()
                if idx == 2:
                    import pdb; pdb.set_trace()
                    print("hehe")
            if epoch == 0 and idx == 2000:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=epochs,
                    eta_min=0.0,
                    last_epoch=-1
                )
            if idx % 1000 == 0:
                tqdm.write(f"Loss: {loss.item()} - Epoch {epoch}")
                torch.cuda.empty_cache()
            if idx % 10000 == 0:
                torch.save(model.state_dict(), f"downloads/weightdecay_gpt1_{epoch}.pth")
        except Exception as e:
            torch.cuda.empty_cache()
            tqdm.write(f"{e}: {x.shape}")
    torch.save(model.state_dict(), f"downloads/weightdecay_gpt1_{epoch}.pth")

    # expect to use cosine scheduler now
    lr_scheduler.step()
