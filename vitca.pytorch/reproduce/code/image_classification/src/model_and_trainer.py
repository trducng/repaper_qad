import torch
from torchvision.utils import make_grid
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import __init__
from image_classification.src.utils import masking_schedule, py2pil
from masked_autoencoding.src.datasets import random_mask
import logging
import re
import dill  # so that torch.save can save lambda functions
import numpy as np
import random
from pathlib import Path
import wandb
from einops import rearrange

LOGGER = logging.getLogger(__name__)


class ModelAndTrainer(torch.nn.Module):
	def __init__(self, cfg, device, phase='train'):
		super(ModelAndTrainer, self).__init__()
		if cfg.experiment.trainer.linear_probing.enabled:
			if cfg.experiment.trainer.linear_probing.use_pretrained_model:
				self.method = 'linear_probing'
			else:
				self.method = 'linear_classification'
		elif cfg.experiment.trainer.fine_tuning.enabled:
			self.method = 'fine_tuning'
		else:
			raise ValueError('Only linear probing or fine-tuning allowed.')

		self.device = device
		if self.method != 'linear_classification':
			self.model, feats_chns = self.inst_model(cfg, device, phase)
		else:
			self.model = None
			feats_chns = cfg.classifier.in_chns
		self.classifier = self.inst_classifier(cfg, feats_chns, device, phase)

		if phase == 'train':
			# Replacing CA-based configs with non-CA equivalents
			if 'CA' not in cfg.pretrained_model._target_ or self.method == 'linear_classification':
				with open_dict(cfg.experiment.trainer.loss):
					cfg.experiment.trainer.loss.pop('_target_', None)
					OmegaConf.update(cfg.experiment.trainer.loss, '_target_', 'image_classification.src.losses.Loss')
			self.loss = instantiate(cfg.experiment.trainer.loss).to(device)
			if self.method == 'linear_probing':
				self.opt = instantiate(cfg.experiment.trainer.linear_probing.opt, params=self.classifier.parameters())
				self.lr_sched = instantiate(cfg.experiment.trainer.linear_probing.lr_sched, optimizer=self.opt)
				self.model.requires_grad_(False)
			elif self.method == 'fine_tuning':
				self.opt = instantiate(
					cfg.experiment.trainer.fine_tuning.opt,
					params=list(self.model.parameters()) + list(self.classifier.parameters())
				)
				self.lr_sched = None
			elif self.method == 'linear_classification':
				self.opt = instantiate(cfg.experiment.trainer.linear_probing.opt, params=self.classifier.parameters())
				self.lr_sched = instantiate(cfg.experiment.trainer.linear_probing.lr_sched, optimizer=self.opt)
		elif phase == 'test':
			# Replacing CA-based configs with non-CA equivalents
			if 'CA' not in cfg.pretrained_model._target_ or self.method == 'linear_classification':
				with open_dict(cfg.experiment.trainer.loss):
					cfg.experiment.trainer.loss.pop('_target_', None)
					OmegaConf.update(cfg.experiment.trainer.loss, '_target_', 'image_classification.src.losses.Loss')
			self.loss = instantiate(cfg.experiment.trainer.loss).to(device)
		elif phase == 'inference':
			# Replacing CA-based configs with non-CA equivalents
			if 'CA' not in cfg.pretrained_model._target_ or self.method == 'linear_classification':
				with open_dict(cfg.loss):
					cfg.loss.pop('_target_', None)
					OmegaConf.update(cfg.loss, '_target_', 'image_classification.src.losses.Loss')
			self.loss = instantiate(cfg.loss).to(device)
		self.train_losses = {
			'cross_entropy_loss': [],
			'reconstruction_loss': [],
			'reconstruction_factor': [],
			'overflow_loss': [],
			'overflow_factor': [],
			'total_loss': []
		}
		self.avg_val_losses = {
			'cross_entropy_loss': [],
			'reconstruction_loss': [],
			'reconstruction_factor': [],
			'overflow_loss': [],
			'overflow_factor': [],
			'total_loss': [],
			'accuracy_top1': [],
			'accuracy_top3': [],
			'accuracy_top5': [],
			'psnr': []
		}
		self.best_avg_val_acc = 1e-8

	def train(self, mode=True):
		if not isinstance(mode, bool):
			raise ValueError('mode is expected to be boolean')
		self.training = mode
		if self.method == 'fine_tuning':
			self.model.train(mode)
		elif self.method == 'linear_probing':
			self.model.eval()
		self.classifier.train(mode)
		return self

	def eval(self):
		return self.train(False)

	def inst_classifier(self, cfg, feats_chns, device, phase='train'):
		if phase == 'train':
			input_size = list(cfg.experiment.input_size.train)
		elif phase == 'test':
			input_size = list(cfg.experiment.input_size.test)
		elif phase == 'inference':
			input_size = list(cfg.input_size)
		if cfg.classifier._target_ == 'image_classification.src.classifiers.linear.Linear':
			if self.method == 'linear_classification':
				feats_chns = feats_chns * input_size[0] * input_size[1]
			classifier = instantiate(cfg.classifier, in_chns=feats_chns).to(device)
		elif cfg.classifier._target_ == 'image_classification.src.classifiers.nonlinear.NonLinear':
			if self.method == 'linear_classification':
				feats_chns = feats_chns * input_size[0] * input_size[1]
			classifier = instantiate(cfg.classifier, in_chns=feats_chns).to(device)
		return classifier

	def inst_model(self, cfg, device, phase='train'):
		if phase == 'train':
			input_size = list(cfg.experiment.input_size.train)
		elif phase == 'test':
			input_size = list(cfg.experiment.input_size.test)
		elif phase == 'inference':
			input_size = list(cfg.input_size)
		if cfg.pretrained_model._target_ == 'masked_autoencoding.src.models.vitca.ViTCA':
			patch_size = cfg.pretrained_model.patch_size
			grid_size = (input_size[0]//patch_size, input_size[1]//patch_size)
			num_patches = grid_size[0] * grid_size[1]

			model = instantiate(cfg.pretrained_model,
								num_patches=num_patches,
								device=device).to(device)

			if cfg.experiment.use_avg_feats:
				feats_chns = model.cell_hidden_chns
			else:
				feats_chns = model.cell_hidden_chns * num_patches

			self.x_pool = []
			self.y_pool = []
			self.z_pool = []
		elif cfg.pretrained_model._target_ == 'masked_autoencoding.src.models.vit.ViT':
			patch_size = cfg.pretrained_model.patch_size
			grid_size = (input_size[0]//patch_size, input_size[1]//patch_size)
			num_patches = grid_size[0] * grid_size[1]

			# Replacing CA-based configs with non-CA equivalents
			if 'cell_in_chns' in cfg.pretrained_model and 'cell_out_chns' in cfg.pretrained_model:
				with open_dict(cfg.pretrained_model):
					in_chns = cfg.pretrained_model.pop('cell_in_chns', None)
					out_chns = cfg.pretrained_model.pop('cell_out_chns', None)
					OmegaConf.update(cfg.pretrained_model, 'in_chns', in_chns)
					OmegaConf.update(cfg.pretrained_model, 'out_chns', out_chns)

			model = instantiate(cfg.pretrained_model,
								num_patches=num_patches,
								device=device).to(device)

			if cfg.experiment.use_avg_feats:
				feats_chns = model.embed_dim
			else:
				feats_chns = model.embed_dim * num_patches
		elif cfg.pretrained_model._target_ == 'masked_autoencoding.src.models.unetca.UNetCA':
			model = instantiate(cfg.pretrained_model, device=device).to(device)

			if cfg.experiment.use_avg_feats:
				feats_chns = model.cell_hidden_dim
			else:
				feats_chns = model.cell_hidden_dim * input_size[0] * input_size[1]

			self.x_pool = []
			self.y_pool = []
			self.z_pool = []
		elif cfg.pretrained_model._target_ == 'masked_autoencoding.src.models.unet.UNet':
			# Replacing CA-based configs with non-CA equivalents
			if 'cell_in_chns' in cfg.pretrained_model and 'cell_out_chns' in cfg.pretrained_model:
				with open_dict(cfg.pretrained_model):
					in_chns = cfg.pretrained_model.pop('cell_in_chns', None)
					out_chns = cfg.pretrained_model.pop('cell_out_chns', None)
					OmegaConf.update(cfg.pretrained_model, 'in_chns', in_chns)
					OmegaConf.update(cfg.pretrained_model, 'out_chns', out_chns)

			model = instantiate(cfg.pretrained_model, device=device).to(device)

			if cfg.experiment.use_avg_feats:
				feats_chns = model.init_features
			else:
				numel = (input_size[0] // 2**model.octaves) * (input_size[1] // 2**model.octaves)
				feats_chns = model.init_features * numel
		return model, feats_chns

	def load_pretrained_model(self, cfg, ckpt_pth=None):
		pretrained_model_path = cfg.experiment.pretrained_model_path.get(str(cfg.dataset.name).lower(), None)
		if pretrained_model_path is None and ckpt_pth is None:
			LOGGER.info('Checkpoint not defined.')
			return False
		if ckpt_pth is not None:
			ckpt_pth = Path(ckpt_pth)
		else:
			ckpt_pth = Path(pretrained_model_path)
		if ckpt_pth.is_file():
			LOGGER.info('Loading pretrained model checkpoint at \'%s\'...', ckpt_pth)
			checkpoint = torch.load(str(ckpt_pth), pickle_module=dill)
			self.model.load_state_dict(checkpoint['state_dict'], strict=False)
			LOGGER.info('Loaded pretrained model checkpoint \'%s\' (at iter %s)', ckpt_pth, checkpoint['iter'])
			return True
		else:
			LOGGER.info('Checkpoint \'%s\' not found.', ckpt_pth)
			return False

	def load_latest_checkpoint(self, cfg, ckpt_dirname):
		# Load latest checkpoint in save dir (cwd) if requested, otherwise randomly initialize
		# check for ckpt_{%d}.pth.tar in current working save dir and pick largest number
		# meant for dealing with preemption and timeouts
		chk_num = lambda pth: int(re.search(r'.*/ckpt_([0-9]+).pth.tar$', str(pth)).group(1))
		chk_pths = sorted(ckpt_dirname.glob('ckpt_[0-9]*.pth.tar'), key=chk_num)
		if bool(chk_pths) and chk_pths[-1].is_file():
			latest_chk_pth = chk_pths[-1]
		else:
			latest_chk_pth = None
		if cfg.experiment.resume_from_latest and latest_chk_pth is not None:
			LOGGER.info('Found checkpoint \'%s\', loading checkpoint and resuming...', latest_chk_pth)
			checkpoint = torch.load(str(latest_chk_pth), pickle_module=dill)
			cfg.experiment.iter.train.start = checkpoint['iter'] + 1
			self.opt.load_state_dict(checkpoint['opt'])
			if self.lr_sched is not None:
				self.lr_sched.load_state_dict(checkpoint['lr_sched'])
			self.train_losses = checkpoint['train_losses']
			self.avg_val_losses = checkpoint['avg_val_losses']
			self.best_avg_val_acc = checkpoint['best_avg_val_acc']
			if 'CA' in cfg.pretrained_model._target_ and self.model is not None:
				self.x_pool = checkpoint['pools']['x']
				self.y_pool = checkpoint['pools']['y']
				self.z_pool = checkpoint['pools']['z']
			if self.model is not None:
				self.model.load_state_dict(checkpoint['state_dicts']['model'])
			self.classifier.load_state_dict(checkpoint['state_dicts']['classifier'])
			LOGGER.info('Loaded checkpoint \'%s\' (at iter %s)', latest_chk_pth, checkpoint['iter'])
			return True
		elif cfg.experiment.resume_from_latest:
			LOGGER.info('No checkpoints found at \'%s\' (not including ckpt_best). Randomly initializing classifier.',
			            ckpt_dirname)
			return False
		else:
			LOGGER.info('Randomly initializing classifier.')
			return False

	def save_checkpoint(self, cfg, i, ckpt_dirname):
		chk_fname = ckpt_dirname / f'ckpt_{i}.pth.tar'
		LOGGER.info('Saving checkpoint at iter %s at %s.', i, chk_fname)
		torch.save({
			'state_dicts': {
				'model': self.model.state_dict() if self.model is not None else None,
				'classifier': self.classifier.state_dict()
			},
			'iter': i,
			'opt': self.opt.state_dict(),
			'lr_sched': self.lr_sched.state_dict() if self.lr_sched is not None else None,
			'last_lr':  self.opt.param_groups[0]['lr'],
			'train_losses': self.train_losses,
			'avg_val_losses': self.avg_val_losses,
			'best_avg_val_acc': self.best_avg_val_acc,
			'pools': {
				'x':self.x_pool,
				'y':self.y_pool,
				'z':self.z_pool
			} if 'CA' in cfg.pretrained_model._target_ and self.model is not None else None
		}, chk_fname, pickle_module=dill)

	def forward(self, cfg, step, x, y, phase='train'):
		if self.method == 'linear_probing':
			with torch.no_grad():
				if 'CA' in cfg.pretrained_model._target_:
					results = self.ca_test_forward(cfg, step, x, y)
				else:
					results = self.reg_test_forward(cfg, step, x, y)
		elif self.method == 'fine_tuning' and cfg.experiment.trainer.fine_tuning.masking_enabled:
			if phase == 'train':
				if 'CA' in cfg.pretrained_model._target_:
					results = self.ca_train_forward(cfg, step, x, y)
				else:
					results = self.reg_train_forward(cfg, step, x, y)
			elif phase == 'validation':
				if 'CA' in cfg.pretrained_model._target_:
					results = self.ca_val_forward(cfg, step, x, y)
				else:
					results = self.reg_val_forward(cfg, step, x, y)
			elif phase == 'test':
				if 'CA' in cfg.pretrained_model._target_:
					results = self.ca_test_forward(cfg, step, x, y)
				else:
					results = self.reg_test_forward(cfg, step, x, y)
			elif phase == 'inference':
				if 'CA' in cfg.pretrained_model._target_:
					results = self.ca_inf_forward(cfg, step, x, y)
				else:
					results = self.reg_inf_forward(cfg, step, x, y)
		elif self.method == 'fine_tuning' and not cfg.experiment.trainer.fine_tuning.masking_enabled:
			if 'CA' in cfg.pretrained_model._target_:
				results = self.ca_test_forward(cfg, step, x, y)
			else:
				results = self.reg_test_forward(cfg, step, x, y)
		elif self.method == 'linear_classification':
			results = {
				'masked_input': x,
				'output_img': x,
				'output_feats': rearrange(x, 'b c h w -> b (c h w)'),
				'ground_truth': {'x': x, 'y': y}
			}

		# Feed feats to classifier
		results['output_logits'] = self.classifier(results['output_feats'])
		return results

	"""
	CA-centric forward functions
	"""
	def ca_train_forward(self, cfg, step, x, y):
		train_size = list(cfg.experiment.input_size.train)
		train_batch_size = cfg.experiment.batch_size.train
		was_sampled = False
		if len(self.z_pool) > train_batch_size and step%2 == 0:
			# Sample from the nca pool, which includes cell states and associated ground truths, every 2nd iter
			x = torch.cat(self.x_pool[:train_batch_size]).to(self.device)
			y = torch.cat(self.y_pool[:train_batch_size]).to(self.device)
			with torch.no_grad():
				z_0 = torch.cat(self.z_pool[:train_batch_size]).to(self.device)
			was_sampled = True
		else:
			p, p_s = masking_schedule(-1,
									  schedule_start=cfg.experiment.masking.train.schedule_start,
									  schedule_end=cfg.experiment.masking.train.schedule_end,
									  max_prob=cfg.experiment.masking.train.max_prob,
									  prob_stages=cfg.experiment.masking.train.prob_stages,
									  max_patch_shape=cfg.experiment.masking.train.max_patch_shape,
									  patch_shape_stages=cfg.experiment.masking.train.patch_shape_stages)
			masked_x = random_mask(x, p=p, mask_type=cfg.experiment.masking.train.type, patch_shape=p_s,
			                       device=self.device)
			with torch.no_grad():
				z_0 = self.model.seed(rgb_in=masked_x, sz=train_size)

		# Iterate nca on masked input for denoising autoencoding
		T = np.random.randint(cfg.experiment.iter.train.ca.min, cfg.experiment.iter.train.ca.max+1)
		if cfg.experiment.trainer.checkpointing.enabled:
			segs = T//2 if T > 1 else 1
			if segs > 1:
				z_0.requires_grad_(True)  # an in-place operation
		else:
			segs = 1
		z_T = self.model(z_0, step_n=T, update_rate=cfg.experiment.iter.train.ca.update_rate, chkpt_segments=segs,
						 attn_size=list(cfg.experiment.attn_size.train))

		# Iterate nca on unmasked input and extract relevant features for linear classifier
		with torch.no_grad():
			z_0_unmasked = self.model.seed(rgb_in=x, sz=train_size)
		z_T_unmasked = self.model(z_0_unmasked, step_n=cfg.experiment.iter.test.ca.value,
								  update_rate=cfg.experiment.iter.test.ca.update_rate,
								  attn_size=list(cfg.experiment.attn_size.test))
		feats = self.model.get_hidden(z_T_unmasked)
		if cfg.experiment.use_avg_feats:
			feats = torch.mean(feats, dim=(-2, -1), keepdim=True)  # spatial pooling
		feats = rearrange(feats, 'b c h w -> b (c h w)')

		return {
			'output_cells': z_T,
			'output_feats': feats,
			'masked_input': self.model.get_rgb_in(z_0),
			'output_img': self.model.get_rgb_out(z_T),
			'output_img_from_unmasked': self.model.get_rgb_out(z_T_unmasked),
			'ground_truth': {'x': x, 'y': y},
			'was_sampled': was_sampled
		}

	def ca_val_forward(self, cfg, step, x, y):
		validation_size = list(cfg.experiment.input_size.val)

		# Use all available kinds of patch shapes and probs
		# TODO: come up with random seed that's not dependent on step since val batch size can affect it
		p, p_s = masking_schedule(-1, max_prob=cfg.experiment.masking.val.max_prob,
								  prob_stages=cfg.experiment.masking.val.prob_stages,
								  max_patch_shape=cfg.experiment.masking.val.max_patch_shape,
								  patch_shape_stages=cfg.experiment.masking.val.patch_shape_stages,
								  random_seed=step)
		masked_x = random_mask(x, p=p, mask_type=cfg.experiment.masking.val.type, patch_shape=p_s,
							   random_seed=cfg.experiment.random_seed, device=self.model.device)
		z_0 = self.model.seed(rgb_in=masked_x, sz=validation_size)

		# Iterate nca on masked input for denoising autoencoding
		z_T = self.model(z_0, step_n=cfg.experiment.iter.val.ca.value,
		                 update_rate=cfg.experiment.iter.val.ca.update_rate,
						 attn_size=list(cfg.experiment.attn_size.val))

		# Iterate nca on unmasked input and extract relevant features for linear classifier
		with torch.no_grad():
			z_0_unmasked = self.model.seed(rgb_in=x, sz=validation_size)
		z_T_unmasked = self.model(z_0_unmasked, step_n=cfg.experiment.iter.test.ca.value,
								  update_rate=cfg.experiment.iter.test.ca.update_rate,
								  attn_size=list(cfg.experiment.attn_size.test))
		feats = self.model.get_hidden(z_T_unmasked)
		if cfg.experiment.use_avg_feats:
			feats = torch.mean(feats, dim=(-2, -1), keepdim=True)  # spatial pooling
		feats = rearrange(feats, 'b c h w -> b (c h w)')

		return {
			'output_cells': z_T,
			'output_feats': feats,
			'masked_input': self.model.get_rgb_in(z_0),
			'output_img': self.model.get_rgb_out(z_T),
			'output_img_from_unmasked': self.model.get_rgb_out(z_T_unmasked),
			'ground_truth': {'x': x, 'y': y},
		}

	def ca_test_forward(self, cfg, step, x, y):
		input_size = list(cfg.experiment.input_size.test)
		if ((self.method == 'linear_probing' and not cfg.experiment.trainer.linear_probing.masking_enabled) or
			self.method == 'fine_tuning'):
			masked_x = x
		else:
			# Use all available kinds of patch shapes and probs
			p, p_s = masking_schedule(-1, max_prob=cfg.experiment.masking.test.max_prob,
									  prob_stages=cfg.experiment.masking.test.prob_stages,
									  max_patch_shape=cfg.experiment.masking.test.max_patch_shape,
									  patch_shape_stages=cfg.experiment.masking.test.patch_shape_stages)
			masked_x = random_mask(x, p=p, mask_type=cfg.experiment.masking.test.type, patch_shape=p_s,
								   device=self.model.device)

		z_0 = self.model.seed(rgb_in=masked_x, sz=input_size)

		# Iterate nca
		z_T = self.model(z_0, step_n=cfg.experiment.iter.test.ca.value,
						 update_rate=cfg.experiment.iter.test.ca.update_rate,
						 attn_size=list(cfg.experiment.attn_size.test))

		# Extract relevant features for linear classifier
		feats = self.model.get_hidden(z_T)
		if cfg.experiment.use_avg_feats:
			feats = torch.mean(feats, dim=(-2, -1), keepdim=True)  # spatial pooling
		feats = rearrange(feats, 'b c h w -> b (c h w)')

		return {
			'output_cells': z_T,
			'output_feats': feats,
			'masked_input': self.model.get_rgb_in(z_0),
			'output_img': self.model.get_rgb_out(z_T),
			'ground_truth': {'x': x, 'y': y},
		}

	def ca_inf_forward(self, cfg, step, x, y):
		input_size = list(cfg.input_size)
		# Use all available kinds of patch shapes and probs if user requested
		if cfg.masking.random:
			p, p_s = masking_schedule(-1, max_prob=cfg.masking.max_prob,
									  prob_stages=cfg.masking.prob_stages,
									  max_patch_shape=cfg.masking.max_patch_shape,
									  patch_shape_stages=cfg.masking.patch_shape_stages)
		else:
			p = cfg.masking.prob
			p_s = cfg.masking.patch_shape
		masked_x = random_mask(x, p=p, mask_type=cfg.masking.type, patch_shape=p_s, device=self.model.device)
		z_0 = self.model.seed(rgb_in=masked_x, sz=input_size)

		# Iterate nca
		z_T = self.model(z_0, step_n=cfg.iter.ca.value, update_rate=cfg.iter.ca.update_rate,
						 attn_size=list(cfg.attn_size))

		# Extract relevant features for linear classifier
		feats = self.model.get_hidden(z_T)
		if cfg.experiment.use_avg_feats:
			feats = torch.mean(feats, dim=(-2, -1), keepdim=True)  # spatial pooling
		feats = rearrange(feats, 'b c h w -> b (c h w)')

		return {
			'output_cells': z_T,
			'output_feats': feats,
			'masked_input': self.model.get_rgb_in(z_0),
			'output_img': self.model.get_rgb_out(z_T),
			'ground_truth': {'x': x, 'y': y},
		}

	"""
	Forward functions for non-CA networks
	"""
	def reg_train_forward(self, cfg, step, x, y):
		p, p_s = masking_schedule(step,
								  schedule_start=cfg.experiment.masking.train.schedule_start,
								  schedule_end=cfg.experiment.masking.train.schedule_end,
								  max_prob=cfg.experiment.masking.train.max_prob,
								  prob_stages=cfg.experiment.masking.train.prob_stages,
								  max_patch_shape=cfg.experiment.masking.train.max_patch_shape,
								  patch_shape_stages=cfg.experiment.masking.train.patch_shape_stages)
		masked_x = random_mask(x, p=p, mask_type=cfg.experiment.masking.train.type, patch_shape=p_s,
							   device=self.device)

		# Forward pass through model with masked input for denoising autoencoding
		out_from_masked = self.model(masked_x, extract_feats=False, method=self.method)

		# Forward pass through model with unmasked input and extract relevant features for linear classifier
		out_from_unmasked, feats = self.model(x, extract_feats=True, method=self.method)
		if cfg.experiment.use_avg_feats:
			feats = torch.mean(feats, dim=(-2, -1), keepdim=True)  # spatial pooling
		feats = rearrange(feats, 'b c h w -> b (c h w)')

		return {
			'masked_input': masked_x,
			'output_img': out_from_masked,
			'output_img_from_unmasked': out_from_unmasked,
			'output_feats': feats,
			'ground_truth': {'x': x, 'y': y},
		}

	def reg_val_forward(self, cfg, step, x, y):
		# use all available kinds of patch shapes and probs
		# TODO: come up with random seed that's not dependent on step since val batch size can affect it
		p, p_s = masking_schedule(-1, max_prob=cfg.experiment.masking.val.max_prob,
								  prob_stages=cfg.experiment.masking.val.prob_stages,
								  max_patch_shape=cfg.experiment.masking.val.max_patch_shape,
								  patch_shape_stages=cfg.experiment.masking.val.patch_shape_stages,
								  random_seed=step)
		masked_x = random_mask(x, p=p, mask_type=cfg.experiment.masking.val.type, patch_shape=p_s,
							   random_seed=cfg.experiment.random_seed, device=self.model.device)

		# Forward pass through model with masked input for denoising autoencoding
		out_from_masked = self.model(masked_x, extract_feats=False, method=self.method)

		# Forward pass through model with unmasked input and extract relevant features for linear classifier
		out_from_unmasked, feats = self.model(x, extract_feats=True, method=self.method)
		if cfg.experiment.use_avg_feats:
			feats = torch.mean(feats, dim=(-2, -1), keepdim=True)  # spatial pooling
		feats = rearrange(feats, 'b c h w -> b (c h w)')

		return {
			'masked_input': masked_x,
			'output_img': out_from_masked,
			'output_img_from_unmasked': out_from_unmasked,
			'output_feats': feats,
			'ground_truth': {'x': x, 'y': y},
		}

	def reg_test_forward(self, cfg, step, x, y):
		if ((self.method == 'linear_probing' and not cfg.experiment.trainer.linear_probing.masking_enabled) or
			self.method == 'fine_tuning'):
			masked_x = x
		else:
			# Use all available kinds of patch shapes and probs
			p, p_s = masking_schedule(-1, max_prob=cfg.experiment.masking.test.max_prob,
									  prob_stages=cfg.experiment.masking.test.prob_stages,
									  max_patch_shape=cfg.experiment.masking.test.max_patch_shape,
									  patch_shape_stages=cfg.experiment.masking.test.patch_shape_stages)
			masked_x = random_mask(x, p=p, mask_type=cfg.experiment.masking.test.type, patch_shape=p_s,
								   device=self.model.device)

		# Forward pass through model and extract relevant features for linear classifier
		out, feats = self.model(masked_x, extract_feats=True, method=self.method)

		if cfg.experiment.use_avg_feats:
			feats = torch.mean(feats, dim=(-2, -1), keepdim=True)  # spatial pooling
		feats = rearrange(feats, 'b c h w -> b (c h w)')

		return {
			'masked_input': masked_x,
			'output_img': out,
			'output_feats': feats,
			'ground_truth': {'x': x, 'y': y},
		}

	def reg_inf_forward(self, cfg, step, x, y):
		# Use all available kinds of patch shapes and probs if user requested
		if cfg.masking.random:
			p, p_s = masking_schedule(-1, max_prob=cfg.masking.max_prob,
									  prob_stages=cfg.masking.prob_stages,
									  max_patch_shape=cfg.masking.max_patch_shape,
									  patch_shape_stages=cfg.masking.patch_shape_stages)
		else:
			p = cfg.masking.prob
			p_s = cfg.masking.patch_shape
		masked_x = random_mask(x, p=p, mask_type=cfg.masking.type, patch_shape=p_s, device=self.model.device)

		# Forward pass through model and extract relevant features for linear classifier
		out, feats = self.model(masked_x, extract_feats=True, method=self.method)

		if cfg.experiment.use_avg_feats:
			feats = torch.mean(feats, dim=(-2, -1), keepdim=True)  # spatial pooling
		feats = rearrange(feats, 'b c h w -> b (c h w)')

		return {
			'masked_input': masked_x,
			'output_img': out,
			'output_feats': feats,
			'ground_truth': {'x': x, 'y': y},
		}

	def update_pools(self, cfg, x, y, z_T, was_sampled):
		"""
		If states were newly created, add new states to nca pool, shuffle, and retain first {pool_size} states.
		If states were sampled from the pool, replace their old states, shuffle, and retain first {pool_size}
		states.
		"""
		pool_size = cfg.experiment.pool_size
		if was_sampled and cfg.experiment.sample_with_replacement:
			train_batch_size = cfg.experiment.batch_size.train
			self.x_pool[:train_batch_size] = torch.split(x, 1)
			self.y_pool[:train_batch_size] = torch.split(y, 1)
			self.z_pool[:train_batch_size] = torch.split(z_T, 1)
		else:
			self.x_pool += list(torch.split(x, 1))
			self.y_pool += list(torch.split(y, 1))
			self.z_pool += list(torch.split(z_T, 1))
		pools = list(zip(self.x_pool, self.y_pool, self.z_pool))
		random.shuffle(pools)
		self.x_pool, self.y_pool, self.z_pool = zip(*pools)
		self.x_pool = list(self.x_pool[:pool_size])
		self.y_pool = list(self.y_pool[:pool_size])
		self.z_pool = list(self.z_pool[:pool_size])

	def log_to_wandb(self, run, step, results, prefix='train', step_prefix='train', scalars_only=False,
	                 images_only=False):
		wandb_log = {}

		if not scalars_only:
			# check if there are images to log
			masked_in = results.pop('input_img', None)
			out_img = results.pop('output_img', None)
			out_img_from_unmasked = results.pop('output_img_from_unmasked', None)
			gt = results.pop('ground_truth_img', None)

			if all(images is not None for images in [masked_in, out_img, gt]):
				# log input images, output images and ground truth
				num = 4  # log first 'num' images
				if not out_img_from_unmasked:
					imgs = torch.cat([masked_in[:num], out_img[:num], gt[:num]], 0)
					caption = 'Top: masked input, Middle: output img, Bottom: ground truth'
				else:
					imgs = torch.cat([masked_in[:num], out_img[:num], out_img_from_unmasked[:num], gt[:num]], 0)
					caption = 'Masked input, output img from masked, output img from unmasked, ground truth'
				imgs = make_grid(imgs, nrow=num, pad_value=1)
				imgs = py2pil(imgs)  # passing my own pil image since wandb normalizes non-pil inputs
				imgs = wandb.Image(imgs, caption=caption)
				wandb_log[f'{prefix}/examples'] = imgs

		# log scalars
		if not images_only:
			for scalar_name, scalar in results.items():
				wandb_log[f'{prefix}/{scalar_name}'] = scalar

		# logging results to wandb
		wandb_log[f'{step_prefix}/step'] = step
		run.log(wandb_log)  # any errors/exceptions will be handled internally

	def update_tracked_scalars(self, scalars, step, phase='train'):
		if phase == 'train':
			for scalar_name, scalar in scalars.items():
				self.train_losses[f'{scalar_name}'].append((step, scalar))
		elif phase == 'validation':
			for scalar_name, scalar in scalars.items():
				self.avg_val_losses[f'{scalar_name}'].append((step, scalar))
