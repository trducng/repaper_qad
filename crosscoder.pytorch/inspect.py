import torch


@torch.no_grad()
def activated_segments(inspector, token_loader, feat_idx, ctx_len, max_segments=None):
    """Return the segments activate a feature

    Args:
        inspector: a crosscoder-hooked inspector
        token_loader: a list of token
    """
    if max_segments is None:
        max_segments = len(token_loader)

    segments = []

    for tokens in token_loader:
        segment = {}
        output = inspector.run(tokens, _new_state=True)
        features = inspector.state.crosscoder.feat  # ctx_len, n_features
        activated = features[:, feat_idx] > 0  # ctx_len
        activated = activated.nonzero().squeeze().tolist()
        for idx in activated:
            segment[idx] = {
                "token": tokens[idx],
                "token_ctx": tokens[max(idx-ctx_len, 0):idx+ctx_len],
                "feature": features[idx, feat_idx],
                "related_features": features[idx].nonzero().squeeze().tolist(),
                "logits": output.logits[idx],
                "logits_ctx": output.logits[max(idx-ctx_len, 0):idx+ctx_len],
            }

    return segments


def steer_model(crosscoder, inspector, token_loaders, segments):
    pass
