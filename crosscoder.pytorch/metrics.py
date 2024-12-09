"""Train a crosscoder, and then hook that crosscoder to a model through an inspector
"""
import numpy as np
import torch
from tqdm import tqdm

from dawnet.inspector import Inspector


def get_feature(crosscoder, inspector: Inspector, dataloader):
    """Get the feature of for the model. Not necessary. If the crosscoder is
    properly implemented, the inspector already contains the feature from crosscoder

    Args:
        crosscoder: the crosscoder model
        inspector: the inspector that can retrieve hidden activation from llm
            already has crosscoder implemented
        dataloader: contains input data
    """
    for tokens in dataloader:
        output, state = inspector.run(tokens, _refresh_state=True)
        feature = state.crosscoder[0]
        yield feature, tokens, output


def sparsity(
    crosscoder, hidden_loader=None, inspector=None, token_loader=None, detail_level=None
):
    """Calculate the percentage of activated features for each token

    Either hidden_act or (inspector and dataloader) should be provided.

    Each input is a list of tokens for a context length.
    Each output is the % of activated features for each token in the context length.

    Args:
        crosscoder: the crosscoder model.
        hidden_act: the hidden activation of the model
        inspector: the inpector model that can be used to get the hidden activation
        dataloader: the dataloader that can be used to get the hidden activation

    Returns:
        sparsity statistics:
            - count: the count of each feature
            - on average, how many features (and %) are activated
            - maximun number of features (and %) are activated
            - minimum number of features (and %) are activated
    """
    outputs = []
    with torch.no_grad():
        if hidden_loader is not None:
            for hidden_feat in tqdm(hidden_loader):
                n, c, l, f = hidden_feat.shape
                hidden_feat = hidden_feat.reshape(n * c, l, f)
                feat = crosscoder.encode(hidden_feat.to(device=crosscoder.device))
                feat = feat.reshape(n, c, -1)
                activated = feat > 0
                outputs.append(activated.sum(dim=2) / feat.shape[2])

    return outputs     # n_batch x ctx_len


def activation_statistics():
    ...


def reconstruction_error():
    ...


def cross_entropy_difference():
    ...


class Metrics:
    def __init__(self):
        self._initiated = False
        self.name = "metrics"

    def initiate(self):
        self._initiated = True

    def update(self, feat):
        pass


class L0(Metrics):

    def __init__(self):
        super().__init__()
        self.name = "sparsity"
        self.result = None
        self.feat_shape = 0

    def initiate(self):
        self._initiated = True
        self.total_inactive = 0
        self.total = 0
        self.n_instances = 0
        self.max_active = 0
        self.min_active = float("inf")

    def update(self, feat: torch.Tensor):
        """feat has shape n x c x f"""
        self.feat_shape = feat.shape[2]
        activated = feat > 0     # n x c x f
        activated_feats = activated.sum(dim=2)    # n x c

        self.total_inactive += (activated_feats == 0).sum().item()
        self.total += activated_feats.sum().item()
        self.n_instances += activated_feats.numel()
        self.max_active = max(self.max_active, activated_feats.max().item())
        self.min_active = min(self.min_active, activated_feats.min().item())

    def finalize(self, result_dict):
        mean = self.total / self.n_instances
        self.result = {
            "total_inactive": self.total_inactive,
            "mean_active": mean,
            "mean_active_pct": mean / self.feat_shape,
            "max_active": self.max_active,
            "max_active_pct": self.max_active / self.feat_shape,
            "min_active": self.min_active,
            "min_active_pct": self.min_active / self.feat_shape,
        }
        result_dict.update(self.result)


class DeadNeurons(Metrics):
    def __init__(self):
        super().__init__()
        self.name = "dead_neurons"
        self.count = None

    def initiate(self):
        super().initiate()
        self.count = None

    def update(self, feat):
        """feat has shape n x c x f"""
        if self.count is None:
            self.count = torch.zeros(feat.shape[2]).to(device=feat.device)
        self.count += (feat != 0).sum(dim=[0, 1])

    def finalize(self, result_dict):
        if self.count is not None:
            dead_neurons = (self.count == 0).sum().item()
            self.result = {
                "total_dead_neurons": dead_neurons,
                "total_neurons": self.count.shape[0],
                "pct_dead_neurons": dead_neurons / self.count.shape[0],
            }
            result_dict.update(self.result)


class ExplainedVariance(Metrics):
    def __init__(self):
        super().__init__()
        self.name = "explained_variance"

    def initiate(self):
        super().initiate()
        self.mean_explained_variance = []
        self.mean_explained_variance_a = []
        self.mean_explained_variabce_b = []
        self.residual = []

    def update(self, x, recon):
        """Each has shape b x l x f"""
        residual = (x.float() - recon.float()) ** 2
        x_var = (x - x.mean(dim=-1, keepdim=True)) ** 2

        explained_variance = (
            1
            - residual.sum(dim=(1, 2)) / (x_var.sum(dim=(1, 2)) + 1e-8)
        ).mean().item()
        explained_variance_a = (
            1
            - residual[:,0,:].sum(dim=1) / (x_var[:,0,:].sum(dim=1) + 1e-8)
        ).mean().item()
        explained_variance_b = (
            1
            - residual[:,1,:].sum(dim=1) / (x_var[:,1,:].sum(dim=1) + 1e-8)
        ).mean().item()

        self.residual.append(residual.cpu().numpy().mean())
        self.mean_explained_variance.append(explained_variance)
        self.mean_explained_variance_a.append(explained_variance_a)
        self.mean_explained_variabce_b.append(explained_variance_b)

    def finalize(self, result_dict):
        result_dict.update({
            "mean_explained_variance": np.mean(self.mean_explained_variance),
            "mean_explained_variance_a": np.mean(self.mean_explained_variance_a),
            "mean_explained_variance_b": np.mean(self.mean_explained_variabce_b),
            "residual": np.mean(self.residual),
        })


class CrossEntropyDifference(Metrics):
    pass


def evaluate(crosscoder, hidden_loader, metrics: list | None = None):
    """Evaluate the model using a list of metrics

    Args:
        crosscoder: the crosscoder model
        hidden_loader: the dataloader that contains the hidden activation
        metrics: the list of metrics to evaluate the model
    """
    if metrics is None:
        metrics = [L0(), DeadNeurons()]

    for metric in metrics:
        metric.initiate()

    with torch.no_grad():
        for hidden_feat in tqdm(hidden_loader):
            n, c, l, f = hidden_feat.shape
            hidden_feat = hidden_feat.reshape(n * c, l, f)
            feat = crosscoder.encode(hidden_feat.to(device=crosscoder.device))  # b x f
            feat = feat.reshape(n, c, -1)
            for metric in metrics:
                metric.update(feat)

    result = {}
    for metric in metrics:
        metric.finalize(result)

    return result


if __name__ == "__main__":
    import torch
    from data import IntermediateStateDataset
    from models import CrossCoderV1
    train_dataset = IntermediateStateDataset(
        path1="/data2/mech/internals/transformer.h.8.npy",
        path2="/data2/mech/internals/transformer.h.9.npy",
    )
    hidden_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    crosscoder = CrossCoderV1.load_from_checkpoint("/home/john/repaper_qad/crosscoder.pytorch/lightning_logs/version_2/checkpoints/epoch=1-step=4038.ckpt").cuda()
    outputs = sparsity(crosscoder, hidden_loader=hidden_loader)
