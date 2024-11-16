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
from models import CrossCoderV1, CrossCoderOp
from metrics import sparsity

model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

crosscoder = CrossCoderV1.load_from_checkpoint("/data2/mech/logs/temp.ckpt").cuda()
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

input_ids = tokenizer.encode("I love the blue sky", return_tensors="pt").cuda()
output, state = inspector.run(input_ids)
