import torch

from einops import rearrange
from einops.layers.torch import Rearrange
from masked_autoencoding.src.utils import pair, xy_meshgrid, vit_positional_encoding, nerf_positional_encoding


class PreNorm(torch.nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.norm = torch.nn.LayerNorm(dim)
		self.fn = fn

	def forward(self, x, **kwargs):
		return self.fn(self.norm(x), **kwargs)


class FeedForward(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
		super().__init__()
		self.net = torch.nn.Sequential(
			torch.nn.Linear(in_dim, hidden_dim),
			torch.nn.GELU(),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(hidden_dim, out_dim),
			torch.nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.net(x)


class Attention(torch.nn.Module):
	def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
		super().__init__()
		inner_dim = head_dim *  heads
		project_out = not (heads == 1 and head_dim == dim)

		self.heads = heads
		self.scale = head_dim ** -0.5

		self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
		self.attend = torch.nn.Softmax(dim=-1)

		self.attn_map = None

		self.to_out = torch.nn.Sequential(
			torch.nn.Linear(inner_dim, dim),
			torch.nn.Dropout(dropout)
		) if project_out else torch.nn.Identity()

	def forward(self, x):
		qkv = self.to_qkv(x).chunk(3, dim=-1)
		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

		dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

		self.attn_maps = attn = self.attend(dots)

		out = torch.matmul(attn, v)
		out = rearrange(out, 'b h n d -> b n (h d)')
		return self.to_out(out)


class Transformer(torch.nn.Module):
	def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
		super().__init__()
		self.layers = torch.nn.ModuleList([])
		for _ in range(depth):
			self.layers.append(torch.nn.ModuleList([
				PreNorm(dim, Attention(dim, heads=heads, head_dim=head_dim, dropout=dropout)),
				PreNorm(dim, FeedForward(dim, mlp_dim, dim, dropout=dropout))
			]))

	def encode(self, x, attn, ff):
		x = attn(x) + x
		x = ff(x) + x
		return x

	def forward(self, x):
		for attn, ff in self.layers:
			x = self.encode(x, attn, ff)
		return x

class ViT(torch.nn.Module):
	def __init__(self, *,
		patch_size=8,
		num_patches=256,
		depth=1,
		heads=1,
		mlp_dim=64,
		dropout=0.,
		in_chns=3,
		out_chns=3,
		embed_input=False,
		embed_dim=32,
		embed_dropout=0.,
		preprocess_fn=None,
		pe_method='vit_handcrafted',
		nerf_pe_basis='sin_cos',
		nerf_pe_max_freq=10,
		device='cpu'):
		super().__init__()
		self.device = device

		self.embed_input = embed_input
		self.pe_method = pe_method
		self.nerf_pe_basis = nerf_pe_basis
		self.nerf_pe_max_freq = nerf_pe_max_freq

		self.patch_height, self.patch_width = pair(patch_size)

		# computing dimensions for layers
		if pe_method == 'nerf_handcrafted':
			self.pe_patch_dim = 2 * 2 * 10 * self.patch_height * self.patch_width
		else:
			self.pe_patch_dim = 0
		self.in_patch_dim = (in_chns * self.patch_height * self.patch_width) + self.pe_patch_dim
		self.out_patch_dim = out_chns * self.patch_height * self.patch_width
		if preprocess_fn is not None:
			self.in_patch_dim *= 4
		if not embed_input:
			embed_dim = self.in_patch_dim
		self.embed_dim = embed_dim

		# rearranging from 2D grid to 1D sequence
		self.vectorize = Rearrange('b c h w -> b (h w) c')

		self.patchify = Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w',
								  p1=self.patch_height, p2=self.patch_width)
		self.unpatchify = Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
									p1=self.patch_height, p2=self.patch_width)

		self.to_embedding = torch.nn.Linear(self.in_patch_dim, embed_dim) if embed_input else None

		if pe_method == 'learned':
			self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches, embed_dim))

		# for convolutional preprocessing of input
		self.preprocess_fn = preprocess_fn

		self.dropout = torch.nn.Dropout(embed_dropout)

		self.transformer = Transformer(embed_dim, depth, heads, embed_dim // heads, mlp_dim, dropout)

		# For fine-tuning for image classification. Normalized pre-logits will be fed to a linear classifier.
		self.norm_prelogits = torch.nn.LayerNorm(embed_dim)

		self.mlp_head = torch.nn.Sequential(
			torch.nn.LayerNorm(embed_dim),
			torch.nn.Linear(embed_dim, self.out_patch_dim)
		)

	def forward(self, imgs, **kwargs):
		if self.pe_method == 'nerf_handcrafted':
			b, _, h, w = imgs.shape
			xy = xy_meshgrid(w, h, -1, 1, -1, 1, b, device=self.device)
			pe = nerf_positional_encoding(xy, self.nerf_pe_max_freq, self.nerf_pe_basis, device=self.device)
			x = torch.cat([imgs, pe], 1)

		# conv preprocessing with ident, sobel_{x,y}, laplacian filters
		if self.preprocess_fn:
			prepped_input = self.preprocess_fn(imgs)
		else:
			prepped_input = imgs

		x = self.patchify(prepped_input)
		patched_img_h, patched_img_w = x.shape[-2:]
		x = self.vectorize(x)

		if self.embed_input:
			x = self.to_embedding(x)

		if self.pe_method == 'vit_handcrafted':
			x += vit_positional_encoding(x.shape[-2], x.shape[-1], device=self.device)
		elif self.pe_method == 'learned':
			x += self.pos_embedding

		x = self.dropout(x)
		if kwargs.get('extract_feats', False) and kwargs.get('method', False) == 'linear_probing':
			feats = self.transformer.layers[0][0].norm(x)
			feats = rearrange(feats, 'b (h w) c -> b c h w', h=patched_img_h, w=patched_img_w)
		x = self.transformer(x)
		if kwargs.get('extract_feats', False) and kwargs.get('method', False) == 'fine_tuning':
			feats = self.norm_prelogits(x)
			feats = rearrange(feats, 'b (h w) c -> b c h w', h=patched_img_h, w=patched_img_w)
		x = self.mlp_head(x)

		x = rearrange(x, 'b (h w) c -> b c h w', h=patched_img_h, w=patched_img_w)
		x = self.unpatchify(x)

		if kwargs.get('extract_feats', False):
			return x, feats
		else:
			return x
