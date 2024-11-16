import math
from typing import Callable

import einops
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dawnet.inspector import Op, Inspector
from dawnet.op import Hook

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
        z_1 = torch.matmul(x[:, 0, :], self.W_enc_1)  # n_batch, n_features
        z_2 = torch.matmul(x[:, 1, :], self.W_enc_2)  # n_batch, n_features
        z = z_1 + z_2
        z = z + self.b_enc  # n_batch, n_features
        z = nn.functional.relu(z)
        return z  # n_batch, n_features

    def decode(self, a):
        """a has shape: n_batch x n_features"""
        z = torch.matmul(a, self.W_dec)  # n_layers, n_batch, n_hidden
        n_layers, n_batch, n_hidden = z.shape
        z = z.view(n_batch, n_layers, n_hidden)
        y = z + self.b_dec  # n_batch, n_layers, n_hidden
        return y

    def forward(self, x):
        """x has shape: n_layers x n_hidden"""
        a_ = self.encode(x)  # n_batch, n_features
        y = self.decode(a_)  # n_batch, n_hidden
        return a_, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_nb):
        act, output = self.forward(batch[0])

        # reconstruction mse term
        reconstruction = self.loss(batch[0], output)

        # regularization term
        W_dec_norm = self.W_dec.norm(p=1, dim=2)  # n_layers, n_features
        W_dec_sum = W_dec_norm.sum(dim=0)  # n_features
        reg = W_dec_sum * act  # n_features
        reg = reg.sum()

        # loss
        loss = reconstruction + 1e-6 * reg
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


class CrossCoderRef(L.LightningModule):
    """Reference from https://github.com/ckkissane/crosscoder-model-diff-replication/blob/main/crosscoder.py"""

    def __init__(self, n_hidden, n_features, dec_init_norm):
        super().__init__(n_hidden, n_features)
        self._hidden_dim = n_hidden
        self._feature_dim = n_features
        self.dtype = torch.float32

        self.W_enc = nn.Parameter(
            torch.empty(2, self._hidden_dim, self._feature_dim, dtype=self.dtype)
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self._feature_dim, 2, self._hidden_dim, dtype=self.dtype),
            )
        )
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * dec_init_norm
        # TODO: what is the purpose of this transpose
        # TODO: look at the role of `dec_init_norm`
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "f l h -> l h f"
        )
        self.b_enc = nn.Parameter(torch.zeros(self._feature_dim, dtype=self.dtype))
        self.b_dec = nn.Parameter(torch.zeros((2, self._hidden_dim), dtype=self.dtype))

    def encode(self, x):
        """x has shape: n_batch x n_layers x hidden_dim"""
        z = torch.einsum("blh,lhf->bf", x, self.W_enc)
        z = z + self.b_enc
        z = torch.nn.functional.relu(z)
        return z

    def decode(self, feat):
        """feat has shape: n_batch x n_features"""
        y = torch.einsum("bf,flh->blh", feat, self.W_dec)
        y = y + self.b_dec
        return y

    def forward(self, x):
        """x has shape: n_layers x hidden_dim"""
        feat = self.encode(x)
        recon = self.decode(feat)
        return feat, recon

    def training_step(self, batch, batch_nb):
        x = batch[0]
        act, recon = self.forward(x)

        diff = (recon - x).pow(2)
        loss_per_batch = einops.reduce(diff, "blh->b", "sum")
        recon_loss = loss_per_batch.mean()

        decoder_norm = self.W_dec.norm(p=1, dim=-1)
        decoder_norm = einops.reduce(decoder_norm, "fl -> f", "sum")
        reg = decoder_norm * act

class CrossCoderOp(Op):
    """Create the crosscoder

    Usage:
        >> crosscoder = CrossCoderV1.load_from_checkpoint("path/to/checkpoint")
        >> inspector.add_op(
            ".",
            CrossCoderOp(
                crosscoder=crosscoder,
                layers={"layer1": None, "layer2": None},
                name="crosscoder",
            )
        )
        >> output, state = inspector.run(tokens)
        >> feat = state.crosscoder
    """

    def __init__(self, crosscoder, layers: dict, name="crosscoder"):
        super().__init__()
        self._crosscoder = crosscoder
        self._layers = layers
        self._ids = []
        self._name = name

    def forward(self, inspector: Inspector, name, module, args, kwargs, output):
        hidden_acts = torch.stack(
            [
                inspector.state.crosscoder_hidden_acts[name].squeeze()
                for name in self._layers
            ],
            dim=1,
        )
        feat = self._crosscoder.forward(hidden_acts.to(self._crosscoder.device))
        setattr(inspector.state, self._name, feat)
        return output

    def add(self, inspector):
        inspector.state.create_section(self._name)
        inspector.state.create_section(f"{self._name}_hidden_acts", default={})
        for name, processor in self._layers.items():
            id_ = inspector.add_op(name, Hook(forward=self.extract_layer(processor)))
            self._ids.append(id_)

    def remove(self, inspector):
        inspector.state.remove_section(self._name)
        inspector.state.remove_section(f"{self._name}_hidden_acts")
        for id_ in self._ids:
            inspector.remove_op(id_)

    def extract_layer(self, processor: None | Callable):
        if processor is None:
            processor = lambda x: x

        def forward(inspector, name, module, args, kwargs, output):
            inspector.state[f"{self._name}_hidden_acts"][name] = processor(output)
            return output

        return forward
