def get_feature(model, inspector=None, dataloader=None):
    """Get the feature of for the model
    """
    ...


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
            - on average, how many features (and %) are activated
            - maximun number of features (and %) are activated
            - minimum number of features (and %) are activated
    """
    ...


def dead_neurons(model, input_data=None) -> dict:
    """Calculate neurons that never get activated
    """
    ...


# def reconstruction_error(model,
