import torch

from einops import rearrange

from masked_autoencoding.src.utils import (
	xy_meshgrid,
	nerf_positional_encoding,
	vit_positional_encoding,
	checkpoint_sequential
)

from masked_autoencoding.src.models.unet import UNet


class UNetCA(torch.nn.Module):
	def __init__(self, *,
		nca_mode=True,
		octaves=2,
		cell_init='constant',
		cell_in_chns=3,
		cell_out_chns=3,
		cell_hidden_chns=32,
		init_features=24,
		pe_method=None,
		nerf_pe_basis='sin_cos',
		nerf_pe_max_freq=5,
		device='cpu'):
		super().__init__()
		self.device = device

		self.nca_mode = nca_mode  # True -> UNet's 3x3 convs are changed to 1x1 except the first
		self.octaves = octaves

		assert cell_init == 'constant' or cell_init == 'random'
		self.cell_init = cell_init
		self.pe_method = pe_method
		self.nerf_pe_basis = nerf_pe_basis
		self.nerf_pe_max_freq = nerf_pe_max_freq

		# computing dimensions for layers
		assert pe_method == 'nerf_handcrafted' or pe_method == 'vit_handcrafted' or pe_method is None
		if pe_method == 'nerf_handcrafted':
			if self.nerf_pe_basis == 'sin_cos' or self.nerf_pe_basis == 'sinc':
				self.cell_pe_dim = 2 * 2 * self.nerf_pe_max_freq
			elif self.nerf_pe_basis == 'raw_xy':
				self.cell_pe_dim = 2
			elif self.nerf_pe_basis == 'sin_cos_xy':
				self.cell_pe_dim = 2 * 2 * self.nerf_pe_max_freq + 2
		else:
			self.cell_pe_dim = 0
		self.cell_in_dim = cell_in_chns
		self.cell_out_dim = cell_out_chns
		self.cell_hidden_dim = cell_hidden_chns
		self.cell_update_dim = self.cell_out_dim + self.cell_hidden_dim
		self.cell_dim = self.cell_pe_dim + self.cell_in_dim + self.cell_out_dim + self.cell_hidden_dim
		self.init_features = init_features

		self.unet = UNet(
			in_chns=self.cell_dim,
			out_chns=self.cell_update_dim,
			init_features=self.init_features,
			octaves=self.octaves,
			nca_mode=self.nca_mode,
			device=self.device
		)

	def f(self, cells, update_rate=0.5):
		_cells = cells

		if self.pe_method == 'vit_handcrafted':
			pe = vit_positional_encoding(cells.shape[-2]*cells.shape[-1], cells.shape[1], device=self.device)
			_cells = _cells + rearrange(pe, 'b (h w) c -> b c h w', h=cells.shape[-2], w=cells.shape[-1])

		update = self.unet(_cells)

		# stochastic cell state update
		if update_rate < 1.0:
			b, _, h, w = cells.shape
			update_mask = (torch.rand(b, 1, h, w, device=self.device)+update_rate).floor()
			updated = cells[:, self.cell_pe_dim+self.cell_in_dim:] + update_mask*update
		else:
			updated = cells[:, self.cell_pe_dim+self.cell_in_dim:] + update
		cells = torch.cat([cells[:, :self.cell_pe_dim+self.cell_in_dim], updated], 1)

		return cells

	def forward(self, cells, step_n=1, update_rate=0.5, chkpt_segments=1, **kwargs):
		if self.training and chkpt_segments > 1:
			# gradient checkpointing to save memory but at the cost of re-computing forward pass
			# during backward pass
			z_star = checkpoint_sequential(self.f, cells, segments=chkpt_segments, seq_length=step_n,
										   update_rate=update_rate)
		else:
			z_star = cells
			for _ in range(step_n):
				z_star = self.f(z_star, update_rate)

		return z_star

	def seed(self, rgb_in, sz):
		# for storing input from external source
		assert sz[0] == rgb_in.shape[2] and sz[1] == rgb_in.shape[3]

		if self.cell_init == 'random':
			# randomly initialize cell output channels between [0,1)
			rgb_out_state = torch.rand(rgb_in.shape[0], self.cell_out_dim, sz[0], sz[1], device=self.device)

			# randomly initialize hidden channels between [-1,1) for inter-cell communication
			hidden_state = torch.rand(rgb_in.shape[0], self.cell_hidden_dim, sz[0], sz[1], device=self.device)*2 - 1
		elif self.cell_init == 'constant':
			# initialize cell output channels with 0.5 (gray image)
			rgb_out_state = torch.zeros(rgb_in.shape[0], self.cell_out_dim, sz[0], sz[1], device=self.device) + 0.5

			# initialize hidden channels with 0 for inter-cell communication
			hidden_state = torch.zeros(rgb_in.shape[0], self.cell_hidden_dim, sz[0], sz[1], device=self.device)

		if self.pe_method == 'nerf_handcrafted':
			xy = xy_meshgrid(sz[0], sz[1], -1, 1, -1, 1, rgb_in.shape[0], device=self.device)
			pe = nerf_positional_encoding(xy, self.nerf_pe_max_freq, self.nerf_pe_basis, device=self.device)
			seed_state = torch.cat([pe, rgb_in, rgb_out_state, hidden_state], 1)
		else:
			seed_state = torch.cat([rgb_in, rgb_out_state, hidden_state], 1)

		return seed_state

	def get_pe_in(self, x):
		return x[:, :self.cell_pe_dim]

	def get_rgb_in(self, x):
		return x[:, self.cell_pe_dim:self.cell_pe_dim+self.cell_in_dim]

	def get_rgb_out(self, x):
		return x[:, self.cell_pe_dim+self.cell_in_dim:
				 self.cell_pe_dim+self.cell_in_dim+self.cell_out_dim]

	def get_rgb(self, x):
		return x[:, self.cell_pe_dim:self.cell_pe_dim+self.cell_in_dim+self.cell_out_dim]
	
	def get_pe_and_rgb(self, x):
		return x[:, :self.cell_pe_dim+self.cell_in_dim+self.cell_out_dim]

	def get_hidden(self, x):
		return x[:, self.cell_pe_dim+self.cell_in_dim+self.cell_out_dim:]
