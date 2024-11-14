"""Train a crosscoder, and then hook that crosscoder to a model through an inspector
"""
import torch


def get_feature(crosscoder, inspector, dataloader):
    """Get the feature of for the model. Not necessary. If the crosscoder is
    properly implemented, the inspector already contains the feature from crosscoder

    Args:
        crosscoder: the crosscoder model
        inspector: the inspector that can retrieve hidden activation from llm
            already has crosscoder implemented
        dataloader: contains input data
    """
    for tokens in dataloader:
        output = inspector._model(tokens)
        feature = inspector.get_hidden_activation(output)
        yield feature


def sparsity(
    crosscoder, hidden_act=None, inspector=None, dataloader=None, detail_level=None
) -> dict:
    """Calculate crosscoder feature sparsity.

    Either hidden_act or (inspector and dataloader) should be provided.

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
    count = None
    with torch.no_grad():
        for input_ids in hidden_act:
            feat = crosscoder.forward(input_ids)
            if count is None:
                count = torch.zeros(feat.shape[1])
            count += torch.sum(feat, dim=0)

    return count     # n_features


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


