from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from models import V2
from data import LoadTokens


VERSION = "V2"
DESC = "Test coupling the base model and the crosscoder in one single implementation"

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

crosscoder = V2(
    n_hidden=768,
    n_features=768 * 16,
    model=model,
    layers=["transformer.h.7", "transformer.h.8"],
    desc=DESC,
).cuda()

train_dataset = LoadTokens(path="/data3/mech/thepile_gpt2_tokenized/train.npy")
val_dataset = LoadTokens(path="/data3/mech/thepile_gpt2_tokenized/val.npy")
# with torch.no_grad():
#     token_ids = torch.from_numpy(train_dataset[:4]).long().cuda()
#     hidden, act, recon = crosscoder(token_ids)

def collate_fn(*args, **kwargs):
    x = torch.from_numpy(np.stack(args[0])).long()
    return x


logger = TensorBoardLogger(save_dir=Path.cwd(), name="logs", version=VERSION)
ckpt_callback = ModelCheckpoint(
    train_time_interval=timedelta(minutes=30),
    save_on_train_epoch_end=True,
)


class RefreshTrainingMetricCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.dead_neurons_tracker.initate()
        pl_module.l0_tracker.initiate()



trainer = L.Trainer(
    accelerator="gpu",
    callbacks=[ckpt_callback, RefreshTrainingMetricCallback()],
    max_epochs=4,
    logger=logger,
    # fast_dev_run=5
    # limit_train_batches=50,
    # limit_val_batches=5,
)

trainer.fit(
    crosscoder,
    train_dataloaders=[
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=16,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    ],
    val_dataloaders=[
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=6,
            collate_fn=collate_fn,
            num_workers=4,
        )
    ],
    # ckpt_path="/data2/mech/logs/V1GDetachWdec0.001/checkpoints/epoch=1-step=3784.ckpt",
)
