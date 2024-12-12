import json
import math
import pickle
import sqlite3
from datetime import timedelta

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import lightning as L
from transformers import AutoTokenizer, AutoModelForCausalLM
from lightning.pytorch.callbacks import ModelCheckpoint

from dawnet import Inspector, op
from dawnet.utils.notebook import run_in_process, is_ready
from dawnet.utils.numpy import NpyAppendArray

from data import IntermediateStateDataset
from models import CrossCoderV1ENormalizeKaimingInitTranspose, CrossCoderOp
from metrics import sparsity

model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

crosscoder = CrossCoderV1ENormalizeKaimingInitTranspose.load_from_checkpoint(
    "/home/john/crosscoders/CrossCoderV1ENormalizeKaimingInitTranspose0.08-2/checkpoints/epoch=2-step=7565.ckpt"
).cuda()
inspector = Inspector(model)

op_id = inspector.add_op(
    "",
    CrossCoderOp(
        crosscoder=crosscoder,
        layers={
            "transformer.h.7": lambda x: x[0],
            "transformer.h.8": lambda x: x[0]
        },
        name="crosscoder",
    ),
)
inspector.add_op(
    "transformer.h.7",
    op.CacheModuleInputOutput(no_output=True, input_getter=lambda a, kw: a[0]),
)
inspector.add_op(
    "transformer.h.8",
    op.CacheModuleInputOutput(no_output=True, input_getter=lambda a, kw: a[0]),
)

input_ids = tokenizer.encode("I love the blue sky", return_tensors="pt").cuda()
output, state = inspector.run(input_ids)
feat, recon = state['crosscoder']
