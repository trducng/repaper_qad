import matplotlib
matplotlib.use("Agg")
import einops
import torch
import torch.nn.functional as f
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange

from models import V2, V2NormalizedInput
from data import LoadTokens

import seaborn as sns
import matplotlib.pyplot as plt

torch.autograd.set_grad_enabled(False)

ckpts = "/data/mech/crosscoders/V2NormalizedInputFixedDecode/checkpoints/epoch=3-step=197049.ckpt"
model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

dataset = LoadTokens(path="/data/mech/crosscoders/test.npy")


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
    """Steer the model when a feature is activated"""
    ...


def get_features_idx(crosscoder, features, corpus, tokens):
    """Get the features of token from the corpus"""
    ...


if __name__ == '__main__':
    crosscoder = V2NormalizedInput.load_from_checkpoint(ckpts, model=model)
    token_ids = torch.from_numpy(dataset[3:4]).long().cuda()

    # overall
    hidden, act, recon = crosscoder(token_ids)
    mse_loss = f.mse_loss(recon, hidden)
    cos = f.cosine_similarity(recon, hidden, dim=-1).mean()
    n_activated = (act > 0).sum() / act.numel()
    print(f"{mse_loss=}, {cos=}, {n_activated=}")

    # decoder
    h = crosscoder.get_hidden(token_ids)
    h = crosscoder.apply_hidden_normalization(h)[:,:,1:,:]
    rh = einops.rearrange(h, "b l c h -> (b c) l h")

    z = einops.einsum(rh, crosscoder.W_enc, "b l h, l h f -> b f")
    z1 = rh[:,0,:] @ crosscoder.W_enc[0]
    z2 = rh[:,1,:] @ crosscoder.W_enc[1]

    zm = z + crosscoder.b_enc
    zm = f.relu(zm)
    mask = zm > 0

    zm1 = z1 * mask
    zm2 = z2 * mask
    b_enc = crosscoder.b_enc * mask
    change = ((zm1 + zm2 + b_enc) - zm).abs().mean()
    print(f"{change=}")

    total = zm - b_enc
    for idx in trange(zm1.shape[0]):
        pct1 = zm1[idx] / total[idx]
        pct2 = zm2[idx] / total[idx]
        n1 = pct1[pct1.isnan().logical_not()]
        n2 = pct2[pct2.isnan().logical_not()]
        diff = n1 - n2

        ax = sns.barplot(diff.cpu().numpy())
        plt.savefig(f"/data/mech/crosscoders/vis/{idx:05}.png")
        ax.clear()
    
    print("Ran")
