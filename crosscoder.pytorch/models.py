import math

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import lightning as L


class CrossCoderV1(L.LightningModule):
    def __init__(self, n_hidden, n_features, n_layers):
        super().__init__()

        self._n_hidden = n_hidden
        self._n_features = n_features
        self._n_layers = n_layers
        self.W_enc_1 = nn.Parameter(torch.empty(n_hidden, n_features))
        self.W_enc_2 = nn.Parameter(torch.empty(n_hidden, n_features))
        self.b_enc = nn.Parameter(torch.empty(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_layers, n_features, n_hidden))
        self.b_dec = nn.Parameter(torch.empty(n_layers, n_hidden))

        self.loss = nn.MSELoss()
        self.reset_parameters()
        self.save_hyperparameters()

    def encode(self, x):
        """x has shape: n_batch x n_layers x n_hidden"""
        z_1 = torch.matmul(x[:,0,:], self.W_enc_1)    # n_batch, n_features
        z_2 = torch.matmul(x[:,1,:], self.W_enc_2)    # n_batch, n_features
        z = z_1 + z_2
        z = z + self.b_enc    # n_batch, n_features
        z = nn.functional.relu(z)
        return z              # n_batch, n_features

    def decode(self, a):
        """a has shape: n_batch x n_features"""
        z = torch.matmul(a, self.W_dec)   # n_layers, n_batch, n_hidden
        n_layers, n_batch, n_hidden = z.shape
        z = z.view(n_batch, n_layers, n_hidden)
        y = z + self.b_dec    # n_batch, n_layers, n_hidden
        return y

    def forward(self, x):
        """x has shape: n_layers x n_hidden"""
        a_ = self.encode(x)   # n_batch, n_features
        y = self.decode(a_)   # n_batch, n_hidden
        return a_, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_nb):
        act, output = self.forward(batch[0])

        # reconstruction mse term
        reconstruction = self.loss(batch[0], output)

        # regularization term
        W_dec_norm = self.W_dec.norm(dim=2)    # n_layers, n_features
        W_dec_sum = W_dec_norm.sum(dim=0)      # n_features
        reg = W_dec_sum * act                  # n_features
        reg = reg.sum()

        # loss
        loss = reconstruction + 1e-4 * reg
        if batch_nb % 10 == 0:
            tqdm.write(f"{loss.item()}, {reconstruction.item()}, {reg.item()}")
        return loss

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_enc_2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

        _, fan_in = nn.init._calculate_fan_in_and_fan_out(self.W_enc_1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_enc, -bound, bound)

        _, fan_in = nn.init._calculate_fan_in_and_fan_out(self.W_dec)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_dec, -bound, bound)
