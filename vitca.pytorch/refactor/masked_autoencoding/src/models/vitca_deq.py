"""
ViT NCA
"""
import torch
import torch.autograd as autograd
from einops import rearrange
from einops.layers.torch import Rearrange

from masked_autoencoding.src.deq import solvers
from masked_autoencoding.src.deq.grad import make_pair, backward_factory

from masked_autoencoding.src.utils import (
	xy_meshgrid,
	vit_positional_encoding,
	nerf_positional_encoding,
	pair,
	LocalizeAttention,
	ExtractOverlappingPatches,
	vec2tensor,
	tensor2vec
)


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
		inner_dim = head_dim * heads
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

	def forward(self, x, localize=None, h=None, w=None):
		qkv = self.to_qkv(x).chunk(3, dim=-1)
		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

		if localize is not None:
			q = rearrange(q, 'b h n d -> b h n 1 d')
			k = localize(k, h, w)  # b h n (attn_height attn_width) d
			v = localize(v, h, w)  # b h n (attn_height attn_width) d

		dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # b h n 1 (attn_height attn_width)

		self.attn_maps = attn = self.attend(dots)  # b h n 1 (attn_height attn_width)

		out = torch.matmul(attn, v)  # b h n 1 d
		out = rearrange(out, 'b h n 1 d -> b n (h d)') if localize else rearrange(out, 'b h n d -> b n (h d)')
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

	def encode(self, x, attn, ff, localize_attn_fn=None, h=None, w=None):
		x = attn(x, localize=localize_attn_fn, h=h, w=w) + x
		x = ff(x) + x
		return x

	def forward(self, x, localize_attn_fn=None, h=None, w=None):
		if self.training and len(self.layers) > 1:
			# gradient checkpointing to save memory but at the cost of re-computing forward pass during backward pass
			funcs = [lambda _x: self.encode(_x, attn, ff, localize_attn_fn, h, w) for attn, ff in self.layers]
			x = torch.utils.checkpoint.checkpoint_sequential(funcs, segments=len(funcs), input=x)
		else:
			for attn, ff in self.layers:
				x = self.encode(x, attn, ff, localize_attn_fn, h, w)
		return x


class ViTCA(torch.nn.Module):
	def __init__(self, *,
		f_solver='anderson',
		f_thresh=60,
		b_solver='broyden',
		b_thresh=36,
		stop_mode='abs',
		eval_factor=1.5,
		ift=True,
		safe_ift=False,
		indexing=[],
		sup_all=True,
		phantom_grad=[1],
		tau=1.0,
		patch_size=8,
		overlapping_patches=True,
		num_patches=256,
		octaves=0,
		depth=1,
		heads=1,
		mlp_dim=64,
		dropout=0.,
		cell_init='constant',
		cell_in_chns=3,
		cell_out_chns=3,
		cell_hidden_chns=9,
		embed_cells=False,
		embed_dim=32,
		embed_dropout=0.,
		localize_attn=None,
		localized_attn_neighbourhood=None,
		pe_method='vit_handcrafted',
		nerf_pe_basis='sin_cos',
		nerf_pe_max_freq=10,
		device='cpu'):
		super().__init__()
		self.device = device

		f_solver = getattr(solvers, f_solver)
		b_solver = getattr(solvers, b_solver)

		assert f_solver in [solvers.anderson, solvers.broyden] and b_solver in [solvers.anderson, solvers.broyden]
		self.f_solver = f_solver
		self.f_thresh = f_thresh
		self.b_solver = b_solver
		self.b_thresh = b_thresh
		self.eval_f_thresh = int(self.f_thresh * eval_factor)
		self.stop_mode = stop_mode
		self.indexing = [*indexing, self.f_thresh]

		# By default, we use the same phantom grad for all corrections.
		# You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
		indexing_pg = make_pair(self.indexing, phantom_grad)
		produce_grad = [backward_factory(grad_type=pg, tau=tau, sup_all=sup_all) for pg in indexing_pg]

		# Enabling ift will replace the last gradient function by IFT.
		if ift:
			produce_grad[-1] = backward_factory(
				grad_type='ift',
				safe_ift=safe_ift,
				b_solver=self.b_solver,
				b_solver_kwargs=dict(threshold=self.b_thresh, stop_mode=self.stop_mode)
			)

		self.produce_grad = produce_grad
		self.hook = None

		assert cell_init == 'constant' or cell_init == 'random'
		self.cell_init = cell_init
		self.localize_attn = localize_attn
		self.localized_attn_neighbourhood = localized_attn_neighbourhood
		self.localize_attn_fn = LocalizeAttention(localized_attn_neighbourhood, device) if localize_attn else None
		self.embed_cells = embed_cells
		self.pe_method = pe_method
		self.nerf_pe_basis = nerf_pe_basis
		self.nerf_pe_max_freq = nerf_pe_max_freq

		self.patch_height, self.patch_width = pair(patch_size)
		self.overlapping_patches = overlapping_patches
		if patch_size == 1:
			self.overlapping_patches = False
		self.extract_overlapping_patches = \
			ExtractOverlappingPatches((self.patch_height, self.patch_width), self.device) \
				if self.overlapping_patches else None

		assert octaves >= 0
		self.octaves = octaves

		# computing dimensions for layers
		if self.pe_method == 'nerf_handcrafted':
			if self.nerf_pe_basis == 'sin_cos' or self.nerf_pe_basis == 'sinc':
				mult = 2 * 2 * self.nerf_pe_max_freq
			elif self.nerf_pe_basis == 'raw_xy':
				mult = 2
			elif self.nerf_pe_basis == 'sin_cos_xy':
				mult = 2 * 2 * self.nerf_pe_max_freq + 2
			self.cell_pe_patch_dim = mult * self.patch_height * self.patch_width \
				if not self.overlapping_patches else mult
		else:
			self.cell_pe_patch_dim = 0
		self.cell_in_patch_dim = cell_in_chns * self.patch_height * self.patch_width \
			if not self.overlapping_patches else cell_in_chns
		self.cell_out_patch_dim = cell_out_chns * self.patch_height * self.patch_width \
			if not self.overlapping_patches else cell_out_chns
		self.cell_hidden_chns = cell_hidden_chns
		self.cell_update_dim = self.cell_out_patch_dim + self.cell_hidden_chns
		self.cell_dim = \
			self.cell_pe_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim + self.cell_hidden_chns \
			if not self.overlapping_patches else \
			self.cell_pe_patch_dim + (cell_in_chns * self.patch_height * self.patch_width) + self.cell_out_patch_dim + self.cell_hidden_chns
		if not embed_cells:
			embed_dim = self.cell_dim

		# rearranging from 2D grid to 1D sequence
		self.rearrange_cells = Rearrange('b c h w -> b (h w) c')

		if not self.overlapping_patches:
			self.patchify = Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w',
									  p1=self.patch_height, p2=self.patch_width)
			self.unpatchify = Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
										p1=self.patch_height, p2=self.patch_width)
		else:
			self.patchify = torch.nn.Identity()
			self.unpatchify = torch.nn.Identity()

		self.cell_to_embedding = torch.nn.Linear(self.cell_dim, embed_dim) if embed_cells else None

		if pe_method == 'learned':
			self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches, embed_dim))

		self.dropout = torch.nn.Dropout(embed_dropout)

		self.transformer = Transformer(embed_dim, depth, heads, embed_dim // heads, mlp_dim, dropout)

		self.mlp_head = torch.nn.Sequential(
			torch.nn.LayerNorm(embed_dim),
			torch.nn.Linear(embed_dim, self.cell_update_dim)
		)

		# don't update cells before first backward pass or else cell grid will have immensely diverged and grads will
		# be large and unhelpful
		self.mlp_head[1].weight.data.zero_()
		self.mlp_head[1].bias.data.zero_()

	def preprocess(self, cells, fn):
		pe_and_rgb_img = self.get_pe_and_rgb(cells)
		feats = fn(pe_and_rgb_img)
		feats_patch = self.patchify(feats)  #SLOW
		hidden = self.get_hidden(cells)
		preprocessed_cells = torch.concat([feats_patch, hidden], 1)
		return preprocessed_cells

	def f(self, cells, update_rate=0.5):
		_cells = cells
		if self.overlapping_patches:
			neighbouring_inputs = self.extract_overlapping_patches(self.get_rgb_in(cells))
			_cells = torch.cat([self.get_pe_in(cells),
								neighbouring_inputs,
								self.get_rgb_out(cells),
								self.get_hidden(cells)], 1)

		x = self.rearrange_cells(_cells)

		if self.embed_cells:
			x = self.cell_to_embedding(x)

		if self.pe_method == 'vit_handcrafted':
			x = x + vit_positional_encoding(x.shape[-2], x.shape[-1], device=self.device)
		elif self.pe_method == 'learned':
			x = x + self.pos_embedding

		x = self.dropout(x)

		x = self.transformer(x, localize_attn_fn=self.localize_attn_fn, h=cells.shape[-2], w=cells.shape[-1])

		# stochastic cell state update
		b, _, h, w = cells.shape
		update = rearrange(self.mlp_head(x), 'b (h w) c -> b c h w', h=h, w=w)
		if update_rate < 1.0:
			update_mask = (torch.rand(b, 1, h, w, device=self.device)+update_rate).floor()
			updated = cells[:, self.cell_pe_patch_dim+self.cell_in_patch_dim:] + update_mask*update
		else:
			updated = cells[:, self.cell_pe_patch_dim+self.cell_in_patch_dim:] + update
		cells = torch.cat([cells[:, :self.cell_pe_patch_dim+self.cell_in_patch_dim], updated], 1)

		return cells

	def _fixed_point_solve(self, deq_func, z_star, **kwargs):
		if 'f_thresh' not in kwargs:
			f_thresh = self.f_thresh

		indexing = self.indexing if self.training else None

		with torch.no_grad():
			result = self.f_solver(
				deq_func,
				x0=z_star,
				threshold=f_thresh, # To reuse previous coarse fixed points
				eps=1e-3 if self.stop_mode == 'abs' else 1e-6,
				stop_mode=self.stop_mode,
				indexing=indexing
			)

			z_star, trajectory = result['result'], result['indexing']
		
		return z_star, trajectory, min(result['rel_trace']), min(result['abs_trace'])

	def forward(self, cells, step_n=1, update_rate=0.5, **kwargs):
		# Preparation. Initial z state is input cell state
		_, c, h, w = cells.shape
		z0 = tensor2vec(cells)
		func = lambda z: tensor2vec(self.f(vec2tensor(z, (c, h, w)), update_rate))

		if self.training:
			_, trajectory, rel_error, abs_error = self._fixed_point_solve(func, z0, **kwargs)

			z_out = []
			for z_pred, produce_grad in zip(trajectory, self.produce_grad):
				z_out += produce_grad(self, z_pred, func)

			z_preds = []
			for z_pred in z_out:
				z_preds.append(vec2tensor(z_pred, (c, h, w)))

			return z_preds[0]
		else:
			# During inference, we directly solve for fixed point
			z_star, _, rel_error, abs_error = self._fixed_point_solve(func, z0, f_thresh=self.eval_f_thresh)
			
			return vec2tensor(z_star, (c, h, w))

	def mitosis(self, cells):
		return cells.repeat_interleave(2, -2).repeat_interleave(2, -1)

	def fusion(self, cells):
		return torch.nn.functional.avg_pool2d(cells, kernel_size=2, stride=2, padding=0)

	def seed(self, rgb_in, sz):
		patch_height, patch_width = (self.patch_height, self.patch_width) if not self.overlapping_patches else (1, 1)

		assert sz[0] % patch_height == 0 and sz[1] % patch_width == 0, \
			'Image dimensions must be divisible by the patch size.'

		size = (sz[0] // patch_height, sz[1] // patch_width)

		# for storing input from external source
		assert sz[0] == rgb_in.shape[2] and sz[1] == rgb_in.shape[3]
		n = rgb_in.shape[0]
		rgb_in_state = self.patchify(rgb_in)

		if self.cell_init == 'random':
			# randomly initialize cell output channels between [0,1)
			rgb_out_state = torch.rand(n, self.cell_out_patch_dim, size[0], size[1], device=self.device)

			# randomly initialize hidden channels between [-1,1) for inter-cell communication
			hidden_state = torch.rand(n, self.cell_hidden_chns, size[0], size[1], device=self.device)*2 - 1
		elif self.cell_init == 'constant':
			# initialize cell output channels with 0.5 (gray image)
			rgb_out_state = torch.zeros(n, self.cell_out_patch_dim, size[0], size[1], device=self.device) + 0.5

			# initialize hidden channels with 0 for inter-cell communication
			hidden_state = torch.zeros(n, self.cell_hidden_chns, size[0], size[1], device=self.device)

		if self.pe_method == 'nerf_handcrafted':
			xy = xy_meshgrid(sz[0], sz[1], -1, 1, -1, 1, n, device=self.device)
			pe = nerf_positional_encoding(xy, self.nerf_pe_max_freq, self.nerf_pe_basis, device=self.device)
			pe = self.patchify(pe)
			seed_state = torch.cat([pe, rgb_in_state, rgb_out_state, hidden_state], 1)
		else:
			seed_state = torch.cat([rgb_in_state, rgb_out_state, hidden_state], 1)

		return seed_state

	def get_pe_in(self, x):
		pe_patch = x[:, :self.cell_pe_patch_dim]
		pe = self.unpatchify(pe_patch)
		return pe

	def get_rgb_in(self, x):
		rgb_patch = x[:, self.cell_pe_patch_dim:self.cell_pe_patch_dim+self.cell_in_patch_dim]
		rgb = self.unpatchify(rgb_patch)
		return rgb

	def get_rgb_out(self, x):
		rgb_patch = x[:, self.cell_pe_patch_dim+self.cell_in_patch_dim:
						 self.cell_pe_patch_dim+self.cell_in_patch_dim+self.cell_out_patch_dim]
		rgb = self.unpatchify(rgb_patch)
		return rgb

	def get_rgb(self, x):
		rgb_patch = x[:, self.cell_pe_patch_dim:self.cell_pe_patch_dim+self.cell_in_patch_dim+self.cell_out_patch_dim]
		rgb = self.unpatchify(rgb_patch)
		return rgb
	
	def get_pe_and_rgb(self, x):
		pe_and_rgb_patch = x[:, :self.cell_pe_patch_dim+self.cell_in_patch_dim+self.cell_out_patch_dim]
		pe_and_rgb = self.unpatchify(pe_and_rgb_patch)
		return pe_and_rgb

	def get_hidden(self, x):
		hidden = x[:, self.cell_pe_patch_dim+self.cell_in_patch_dim+self.cell_out_patch_dim:]
		return hidden
