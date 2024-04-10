import torch
from typing import OrderedDict


class ConvLeakyReLU(torch.nn.Module):
	def __init__(self, in_chns, out_chns, kernel_size=3):
		super().__init__()
		self.block = torch.nn.Sequential(
			torch.nn.Conv2d(in_chns, out_chns, kernel_size=kernel_size, padding='same'),
			torch.nn.LeakyReLU(0.1)
		)
	
	def forward(self, x):
		return self.block(x)


class UNet(torch.nn.Module):
	def __init__(self, in_chns=3, out_chns=3, init_features=48, octaves=5, nca_mode=False, device='cpu'):
		super().__init__()
		self.in_chns = in_chns
		self.out_chns = out_chns
		self.init_features = init_features
		self.octaves = octaves
		self.nca_mode = nca_mode
		self.device = device

		# Initial convolution (perception stage if used as nca)
		self.layers = torch.nn.ModuleDict(OrderedDict({
			'enc_conv0': ConvLeakyReLU(in_chns, init_features, 3)
		}))

		ksize = 1 if nca_mode else 3

		# Downsampling stage
		for i in range(1, octaves+1, 1):
			self.layers.update(OrderedDict({
				f'enc_conv{i}': ConvLeakyReLU(init_features, init_features, ksize),
				f'pool{i}': torch.nn.MaxPool2d(kernel_size=2, stride=2)
			}))

		# Bottleneck stage
		self.layers.update(OrderedDict({
			f'enc_conv{octaves+1}': ConvLeakyReLU(init_features, init_features, ksize)
		}))

		# Upsampling stage
		for i in range(octaves, 1, -1):
			mult = 2 if i == octaves else 3
			self.layers.update(OrderedDict({
				# <- upsample nearest x2 ->
				# <- concat with pool{i-1} ->
				f'dec_conv{i}a': ConvLeakyReLU(init_features*mult, init_features*2, ksize),
				f'dec_conv{i}b': ConvLeakyReLU(init_features*2, init_features*2, ksize)
			}))

		# Final stage
		mult = 2 if octaves > 1 else 1
		self.layers.update(OrderedDict({
			# <- upsample nearest x2 ->
			# <- concat with input ->
			'dec_conv1a': ConvLeakyReLU(init_features*mult+in_chns, 64, ksize),
			'dec_conv1b': ConvLeakyReLU(64, 32, ksize),
			'dec_conv1c': torch.nn.Conv2d(32, out_chns, ksize, padding='same')  # no activation
		}))

	def forward(self, x, **kwargs):
		skip_connects = {'input': x}

		# Initial stage
		x = self.layers['enc_conv0'](x)

		# Downsampling stage
		for i in range(1, self.octaves+1, 1):
			x = self.layers[f'enc_conv{i}'](x)
			x = self.layers[f'pool{i}'](x)
			skip_connects[f'pool{i}'] = x

		# Bottleneck stage
		x = self.layers[f'enc_conv{self.octaves+1}'](x)

		if kwargs.get('extract_feats', False):
			feats = x

		# Upsampling stage
		for i in range(self.octaves, 1, -1):
			x = torch.nn.functional.interpolate(x, scale_factor=2., mode='nearest')
			x = torch.cat([x, skip_connects[f'pool{i-1}']], dim=1)
			x = self.layers[f'dec_conv{i}a'](x)
			x = self.layers[f'dec_conv{i}b'](x)

		# Final stage
		if self.octaves > 0:
			x = torch.nn.functional.interpolate(x, scale_factor=2., mode='nearest')
		x = torch.cat([x, skip_connects['input']], dim=1)
		x = self.layers['dec_conv1a'](x)
		x = self.layers['dec_conv1b'](x)
		x = self.layers['dec_conv1c'](x)

		if kwargs.get('extract_feats', False):
			return x, feats
		else:
			return x
