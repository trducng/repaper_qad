"""Train a crosscoder, and then hook that crosscoder to a model through an inspector
"""
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


def dead_neurons(model, input_data=None) -> dict:
    """Calculate neurons that never get activated
    """
    ...


def activation_statistics():
    ...


def reconstruction_error():
    ...


def cross_entropy_difference():
    ...


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
