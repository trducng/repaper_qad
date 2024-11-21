from datetime import timedelta
from pathlib import Path

import torch
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import TensorBoardLogger


from data import IntermediateStateDataset
from models import CrossCoderV1, CrossCoderRef, CrossCoderV1A, CrossCoderV1B, CrossCoderV1C, CrossCoderV1DUseKamingInitTranspose, CrossCoderV1ENormalizeKaimingInitTranspose

VERSION = "CrossCoderV1ENormalizeKaimingInitTranspose0.12"
DESC = "Normalize the crosscoder that has kaiming init transposed, with the norm to be 0.12"

if not VERSION or not DESC:
    raise ValueError("Please set VERSION and DESC")


def collate_fn(*args, **kwargs):
    x = torch.from_numpy(np.concatenate(args[0]))
    return x


# model = CrossCoderV1A(n_features=768 * 16, n_hidden=768, n_layers=2, desc=DESC)
# model = CrossCoderRef(n_features=768 * 16, n_hidden=768, dec_init_norm=0.18, desc=DESC)
# model = CrossCoderV1C(n_features=768 * 16, n_hidden=768, n_layers=2)
model = CrossCoderV1ENormalizeKaimingInitTranspose(n_features=768 * 16, n_hidden=768, n_layers=2, desc=DESC, dec_init_norm=0.12)
logger = TensorBoardLogger(save_dir=Path.cwd(), name="logs", version=VERSION)
ckpt_callback = ModelCheckpoint(train_time_interval=timedelta(minutes=30))
# tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path.cwd(), name="logs")
train_dataset = IntermediateStateDataset(
    path1="/data2/mech/internals/transformer.h.8.npy",
    path2="/data2/mech/internals/transformer.h.9.npy",
)



trainer = L.Trainer(
    accelerator="gpu", callbacks=[ckpt_callback], max_epochs=3, logger=logger,
    # fast_dev_run=5
    # limit_train_batches=50,
    # limit_val_batches=5,
)
trainer.fit(
    model,
    train_dataloaders=[
        torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
    ],
    val_dataloaders=[
        torch.utils.data.DataLoader(train_dataset, batch_size=16)
    ]
)
