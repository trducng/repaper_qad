from datetime import timedelta
from pathlib import Path

import torch
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers


from data import IntermediateStateDataset
from models import CrossCoderV1, CrossCoderRef


def collate_fn(*args, **kwargs):
    x = torch.from_numpy(np.concatenate(args[0]))
    return x


ckpt_callback = ModelCheckpoint(train_time_interval=timedelta(minutes=30))
# tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path.cwd(), name="logs")
train_dataset = IntermediateStateDataset(
    path1="/data2/mech/internals/transformer.h.8.npy",
    path2="/data2/mech/internals/transformer.h.9.npy",
)


# model = CrossCoderV1(n_features=768 * 16, n_hidden=768, n_layers=2)
model = CrossCoderRef(n_features=768 * 16, n_hidden=768, dec_init_norm=0.08)
trainer = L.Trainer(
    accelerator="gpu", callbacks=[ckpt_callback], max_epochs=2
)
trainer.fit(
    model,
    train_dataloaders=[
        torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
    ],
)
