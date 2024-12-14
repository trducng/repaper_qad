import math
from typing import Callable

import einops
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dawnet.inspector import Op, Inspector
from dawnet import op
from dawnet.op import Hook

import lightning as L

from metrics import L0, DeadNeurons, ExplainedVariance, ExplainedVarianceV2


class CrossCoderV1(L.LightningModule):
    def __init__(self, n_hidden, n_features, n_layers, desc):
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
        self.feat_metrics = [L0(), DeadNeurons()]
        self.recon_metrics = [ExplainedVariance()]

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
            self.log("loss", loss.item())
            self.log("recon", reconstruction.item())
            self.log("reg", reg.item())
        return loss

    def on_validation_epoch_start(self):
        for metric in self.feat_metrics:
            metric.initiate()
        for metric in self.recon_metrics:
            metric.initiate()

    def validation_step(self, batch, batch_nb):
        """Work on the validation"""
        n, c, l, f = batch.shape
        hidden_feat = batch.reshape(n * c, l, f).to(self.device)
        feat, recon = self.forward(hidden_feat)
        feat = feat.reshape(n, c, -1)
        for metric in self.feat_metrics:
            metric.update(feat)
        for metric in self.recon_metrics:
            metric.update(hidden_feat, recon)

    def on_validation_epoch_end(self):
        result = {}
        for metric in self.feat_metrics:
            metric.finalize(result)
        for metric in self.recon_metrics:
            metric.finalize(result)
        self.log_dict(result)
        print(result)

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
    """Reference from https://github.com/ckkissane/crosscoder-model-diff-replication/blob/main/crosscoder.py

    Note:
        - Varying `dec_init_norm` from 0.03, 1.0 greatly increases the number of
        dead neurons. So, weight initialization is crucial to obtain good output.
    """

    def __init__(self, n_hidden, n_features, dec_init_norm, desc):
        super().__init__()
        self._hidden_dim = n_hidden
        self._feature_dim = n_features
        self._dtype = torch.float32

        self.W_enc = nn.Parameter(
            torch.empty(2, self._hidden_dim, self._feature_dim, dtype=self._dtype)
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self._feature_dim, 2, self._hidden_dim, dtype=self._dtype),
            )
        )
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * dec_init_norm
        )
        # TODO: what is the purpose of this transpose of making it balance. Virtually nothing?
        # TODO: look at the role of `dec_init_norm`
        self.W_enc.data = einops.rearrange(self.W_dec.data.clone(), "f l h -> l h f")
        self.b_enc = nn.Parameter(torch.zeros(self._feature_dim, dtype=self._dtype))
        self.b_dec = nn.Parameter(torch.zeros((2, self._hidden_dim), dtype=self._dtype))
        self.feat_metrics = [L0(), DeadNeurons()]
        self.recon_metrics = [ExplainedVariance()]
        self.save_hyperparameters()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def encode(self, x):
        """x has shape: n_batch x n_layers x hidden_dim"""
        z = torch.einsum("b l h, l h f -> b f", x, self.W_enc)
        z = z + self.b_enc
        z = torch.nn.functional.relu(z)
        return z

    def decode(self, feat):
        """feat has shape: n_batch x n_features"""
        y = torch.einsum("b f, f l h -> b l h", feat, self.W_dec)
        y = y + self.b_dec
        return y

    def forward(self, x):
        """x has shape: n_layers x hidden_dim"""
        feat = self.encode(x)
        recon = self.decode(feat)
        return feat, recon

    def training_step(self, batch, batch_nb):
        """The main difference is about sum and mean"""
        x = batch[0]
        act, recon = self.forward(x)

        diff = (recon - x).pow(2)
        loss_per_batch = einops.reduce(diff, "b l h -> b", "sum")
        recon_loss = loss_per_batch.mean()

        W_dec_norm = self.W_dec.norm(p=1, dim=2)  # n_features, n_layers
        W_dec_norm = einops.reduce(W_dec_norm, "f l -> f", "sum")  # n_features
        reg = W_dec_norm * act  # n_feat * n_batch, n_feat -> n_batch, n_feat
        reg = reg.sum(dim=1)  # n_batch
        reg = reg.mean()

        loss = recon_loss + reg
        if batch_nb % 10 == 0:
            self.log("loss", loss.item())
            self.log("recon", recon_loss.item())
            self.log("reg", reg.item())

        return loss

    def on_validation_epoch_start(self):
        for metric in self.feat_metrics:
            metric.initiate()
        for metric in self.recon_metrics:
            metric.initiate()

    def validation_step(self, batch, batch_nb):
        """Work on the validation"""
        n, c, l, f = batch.shape
        hidden_feat = batch.reshape(n * c, l, f).to(self.device)
        feat, recon = self.forward(hidden_feat)
        feat = feat.reshape(n, c, -1)
        for metric in self.feat_metrics:
            metric.update(feat)
        for metric in self.recon_metrics:
            metric.update(hidden_feat, recon)

    def on_validation_epoch_end(self):
        result = {}
        for metric in self.feat_metrics:
            metric.finalize(result)
        for metric in self.recon_metrics:
            metric.finalize(result)
        self.log_dict(result)
        print(result)


class CrossCoderV1A(CrossCoderV1):
    """Similar to CrossCoderV1 but with different loss calculation A

    Result:
        - Running slower than CrossCoderRef
        - But have much sparser representation, on average, it has 79.3 active
          features, while CrossCoderRef has 200.6 active features

    Next step: try to balance the encoder and decoder weight.
    """

    def training_step(self, batch, batch_nb):
        x = batch[0]
        act, recon = self.forward(x)

        # reconstruction mse term
        diff = (recon - x).pow(2)
        loss_per_batch = einops.reduce(diff, "b l h -> b", "sum")
        recon_loss = loss_per_batch.mean()

        # regularization term
        W_dec_norm = self.W_dec.norm(p=1, dim=2)  # n_layers, n_features
        W_dec_sum = W_dec_norm.sum(dim=0)  # n_features
        reg = W_dec_sum * act  # n_batch, n_features
        reg = reg.sum(dim=1)  # n_batch
        reg = reg.mean()

        # loss
        loss = recon_loss + reg
        if batch_nb % 10 == 0:
            self.log("loss", loss.item())
            self.log("recon", recon_loss.item())
            self.log("reg", reg.item())
        return loss


class CrossCoderV1B(CrossCoderV1):
    """Balance the encoder and decoder weight, from the original version.

    Result:
        - All the features become zero

    It seems transposing the encoder and decoder make no difference. Yes, why should it
    be when the transpose operation is not a reversible operation.
    """

    def __init__(self, n_hidden, n_features, n_layers, desc):
        super().__init__(n_hidden, n_features, n_layers, desc)
        self.W_enc_1.data = self.W_dec.data[0].T.clone()
        self.W_enc_2.data = self.W_dec.data[1].T.clone()


class CrossCoderV1C(CrossCoderV1A):
    """Balance the encoder and decoder weight, from improved loss calculation A version"""

    def __init__(self, n_hidden, n_features, n_layers, desc):
        super().__init__(n_hidden, n_features, n_layers, desc)
        self.W_enc_1.data = self.W_dec.data[0].T.clone()
        self.W_enc_2.data = self.W_dec.data[1].T.clone()


class CrossCoderV1DUseKamingInitTranspose(CrossCoderV1A):
    """Surprisingly it has 5000 dead neurons"""

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc_1.T, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W_enc_2.T, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W_dec.T, nonlinearity="linear")

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_enc_1.T)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_enc, -bound, bound)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_dec[:, 0, :].T)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_dec, -bound, bound)


class CrossCoderV1ENormalizeKaimingInitTranspose(CrossCoderV1DUseKamingInitTranspose):
    """
    Result:
        - Surprisingly, it has only 65 dead neurons, with quite good L0 (0.0228
          activated)

    It seems to be important to normalize the weight, so that it allow for a good norm.
    """

    def __init__(self, n_hidden, n_features, n_layers, desc, dec_init_norm):
        self.dec_init_norm = dec_init_norm
        super().__init__(n_hidden, n_features, n_layers, desc)
        self.save_hyperparameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.W_dec.data = (
            self.W_dec.data
            / self.W_dec.data.norm(dim=1, keepdim=True)
            * self.dec_init_norm
        )


class V1FNoWdecInReg(CrossCoderV1ENormalizeKaimingInitTranspose):
    def training_step(self, batch, batch_nb):
        x = batch[0]
        act, recon = self.forward(x)

        # reconstruction mse term
        diff = (recon - x).pow(2)
        loss_per_batch = einops.reduce(diff, "b l h -> b", "sum")
        recon_loss = loss_per_batch.mean()

        # regularization term
        reg = act.sum(dim=1)  # n_batch
        reg = reg.mean()

        # loss
        loss = recon_loss + reg
        if batch_nb % 10 == 0:
            self.log("loss", loss.item())
            self.log("recon", recon_loss.item())
            self.log("reg", reg.item())
        return loss


class V1GDetachWdec(CrossCoderV1ENormalizeKaimingInitTranspose):
    def __init__(
        self, n_hidden, n_features, n_layers, desc, dec_init_norm, lmb=1.0, lr=1e-4
    ):
        super().__init__(n_hidden, n_features, n_layers, desc, dec_init_norm)
        self.lmb = lmb
        self.lr = lr

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_nb):
        x = batch[0]
        act, recon = self.forward(x)

        # reconstruction mse term
        diff = (recon - x).pow(2)
        loss_per_batch = einops.reduce(diff, "b l h -> b", "sum")
        recon_loss = loss_per_batch.mean()

        # regularization term
        W_dec = self.W_dec.detach()
        W_dec_norm = W_dec.norm(p=2, dim=2)  # n_layers, n_features
        W_dec_sum = W_dec_norm.sum(dim=0)  # n_features
        reg = W_dec_sum * act  # n_batch, n_features
        reg = reg.sum(dim=1)  # n_batch
        reg = reg.mean()

        # loss
        loss = recon_loss + self.lmb * reg
        if batch_nb % 10 == 0:
            self.log("loss", loss.item())
            self.log("recon", recon_loss.item())
            self.log("reg", reg.item())
        return loss


class V2(L.LightningModule):
    """This implement starts from CrossCoderV1A, and add in the original model"""
    def __init__(self, n_hidden, n_features, model, layers, desc):
        super().__init__()

        self._n_hidden = n_hidden
        self._n_features = n_features
        self._n_layers = len(layers)
        self.layers = layers
        self.W_enc = nn.Parameter(torch.empty(self._n_layers, n_hidden, n_features))
        self.b_enc = nn.Parameter(torch.empty(n_features))
        self.W_dec = nn.Parameter(torch.empty(self._n_layers, n_features, n_hidden))
        self.b_dec = nn.Parameter(torch.empty(self._n_layers, n_hidden))

        self.loss = nn.MSELoss()
        self.reset_parameters()
        self.save_hyperparameters(ignore=["model"])
        self.feat_metrics = [L0(), DeadNeurons()]
        self.recon_metrics = [ExplainedVarianceV2()]
        self.inspector = Inspector(model)
        for name in layers:
            self.inspector.add_op(
                name,
                op.CacheModuleInputOutput(no_input=True, output_getter=lambda x: x[0]),
            )
        # training time metrics
        self.dead_neurons_tracker = DeadNeurons()
        self.dead_neurons_tracker.initiate()
        self.l0_tracker = L0()
        self.l0_tracker.initiate()

    def get_hidden(self, x):
        """
        x has shape: n_batch x ctx_len
        output has shape: n_batch x n_layers x ctx_len x n_hidden
        """
        with torch.no_grad():
            _, state = self.inspector.run(x)
            acts = torch.stack([state["output"][name] for name in self.layers], dim=1)
        return acts

    def encode(self, x):
        """x has shape: n_batch x n_layers x n_hidden"""
        z = einops.einsum(x, self.W_enc, "b l h, l h f -> b f")
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
        """x has shape: n_batch x ctx_len
            n_layers x n_hidden"""
        hidden = self.get_hidden(x)   # n_batch, n_layers, ctx_len, n_hidden
        reshaped_hidden = einops.rearrange(hidden, "b l c h -> (b c) l h") # n_batch, layers, hidden
        a_ = self.encode(reshaped_hidden)  # n_batch, n_features
        y = self.decode(a_)  # n_batch, n_layers, n_hidden
        y = einops.rearrange(y, "(b c) l h -> b l c h", c=hidden.shape[2])  # n_batch, layers, ctx_len, hidden
        return hidden, a_, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_nb):
        x = batch[0]
        hidden, act, recon = self.forward(x)

        # reconstruction mse term
        diff = (recon - hidden).pow(2)
        loss_per_batch = einops.reduce(diff, "b l c h -> (b c)", "sum")
        recon_loss = loss_per_batch.mean()

        # regularization term
        W_dec_norm = self.W_dec.norm(p=1, dim=2)  # n_layers, n_features
        W_dec_sum = W_dec_norm.sum(dim=0)  # n_features
        reg = W_dec_sum * act  # n_batch, n_features
        reg = reg.sum(dim=1)  # n_batch
        reg = reg.mean()

        # loss
        loss = recon_loss + reg
        if batch_nb % 10 == 0:
            self.log("loss", loss.item())
            self.log("recon", recon_loss.item())
            self.log("reg", reg.item())

        with torch.no_grad():
            self.dead_neurons_tracker.update(act)
            if batch_nb % 100 == 0:
                self.l0_tracker.update(act)

        if batch_nb % 1000 == 0:
            result_dict = {}
            self.dead_neurons_tracker.finalize(result_dict)
            self.l0_tracker.finalize(result_dict)
            new_result_dict = {f"train/{k}": v for k, v in result_dict.items()}
            del result_dict
            self.log_dict(new_result_dict)
            del new_result_dict

        if batch_nb % 12000 == 0:
            # Restart the tracker
            self.dead_neurons_tracker.initiate()


        return loss

    def on_validation_epoch_start(self):
        for metric in self.feat_metrics:
            metric.initiate()
        for metric in self.recon_metrics:
            metric.initiate()

    def validation_step(self, batch, batch_nb):
        """Work on the validation"""
        n, c = batch.shape
        hidden, feat, recon = self.forward(batch)
        feat = feat.reshape(n, c, -1)
        for metric in self.feat_metrics:
            metric.update(feat)
        for metric in self.recon_metrics:
            metric.update(hidden, recon)

    def on_validation_epoch_end(self):
        result = {}
        for metric in self.feat_metrics:
            metric.finalize(result)
        for metric in self.recon_metrics:
            metric.finalize(result)
        self.log_dict(result)
        print(result)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

        _, fan_in = nn.init._calculate_fan_in_and_fan_out(self.W_enc)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_enc, -bound, bound)

        _, fan_in = nn.init._calculate_fan_in_and_fan_out(self.W_dec)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_dec, -bound, bound)


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
                inspector.state["crosscoder_hidden_acts"][name].squeeze()
                for name in self._layers
            ],
            dim=1,
        )
        feat = self._crosscoder.forward(hidden_acts.to(self._crosscoder.device))
        inspector.state[self._name] = feat
        return output

    def add(self, inspector):
        inspector.state.register(self._name)
        inspector.state.register(f"{self._name}_hidden_acts", default={})
        for name, processor in self._layers.items():
            id_ = inspector.add_op(name, Hook(forward=self.extract_layer(processor)))
            self._ids.append(id_)

    def remove(self, inspector):
        inspector.state.deregister(self._name)
        inspector.state.deregister(f"{self._name}_hidden_acts")
        for id_ in self._ids:
            inspector.remove_op(id_)

    def extract_layer(self, processor: None | Callable):
        if processor is None:
            processor = lambda x: x

        def forward(inspector, name, module, args, kwargs, output):
            inspector.state[f"{self._name}_hidden_acts"][name] = processor(output)
            return output

        return forward
