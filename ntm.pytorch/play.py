import torch

import toys
from lstm_memory import LSTM_NTM

lstm = LSTM_NTM(
    input_size=10, hidden_size=20, memory_size=5, memory_dim=6,
    n_read_heads=2, n_write_heads=3, shift=1)

task = toys.Copy(shape=10)
batch, width = task.get_batch(1)
batch = torch.Tensor(batch).transpose(0, 1).contiguous()
output, hidden = lstm(batch)
