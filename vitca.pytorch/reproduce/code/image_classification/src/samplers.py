import random
import torch
from torch.utils.data.sampler import Sampler


class BalancedBatchSampler(Sampler):
	"""
	BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
	Returns batches of size n_classes * n_samples

	Adapted from https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4
	"""
	def __init__(
		self,
		dataset,
		episodes,
		episode_length,
		num_classes=None,
		samples_per_class=1,
		classes=None,
		shuffle=False):
		if num_classes is not None and classes is not None:
			raise ValueError('num_classes and classes options are mutually exclusive')
		self.total_num_classes = len(dataset.classes)
		if (classes is None and
			(not isinstance(num_classes, int) or
			 num_classes <= 0 or
			 num_classes > self.total_num_classes)):
			raise ValueError(f'num_classes should be between 1 and {self.total_num_classes}, '
							 f'but got num_classes={num_classes}')
		if not isinstance(samples_per_class, int) or samples_per_class <= 0:
			raise ValueError(
				f'samples_per_class should be a positive integer value, '
				f'but got samples_per_class={samples_per_class}'
			)
		if dataset.targets is None:
			raise ValueError('dataset must have a defined targets attribute')

		self.shuffle = shuffle

		self.targets = torch.LongTensor(dataset.targets)
		self.class_to_idxs = {
			class_: torch.where(self.targets == dataset.class_to_idx[class_])[0] for class_ in dataset.classes
		}

		if shuffle:
			for class_, idxs in self.class_to_idxs.items():
				perm = torch.randperm(len(idxs))
				self.class_to_idxs[class_] = self.class_to_idxs[class_][perm]

		self.used_class_idxs_count = {class_: 0 for class_ in self.class_to_idxs.keys()}
		if classes is not None:
			self.classes = classes
			self.classes_predefined = True
		else:
			self.classes = dataset.classes
			self.classes_predefined = False
		self.num_classes = num_classes if num_classes is not None else len(self.classes)
		self.samples_per_class = samples_per_class
		self.episodes = episodes
		self.episode_length = episode_length
		self.dataset = dataset
		self.batch_size = self.samples_per_class * self.num_classes

	def __iter__(self):
		episode = 1
		count = 0
		classes = self.classes
		while episode <= self.episodes:
			if not self.classes_predefined:
				classes = random.sample(self.classes, k=self.num_classes)
			batch = []
			for class_ in classes:
				batch.extend(
					self.class_to_idxs[class_][
						self.used_class_idxs_count[class_]:self.used_class_idxs_count[class_]+self.samples_per_class
					].tolist()
				)
				self.used_class_idxs_count[class_] += self.samples_per_class
				if self.used_class_idxs_count[class_]+self.samples_per_class > len(self.class_to_idxs[class_]):
					if self.shuffle:
						perm = torch.randperm(len(self.class_to_idxs[class_]))
						self.class_to_idxs[class_] = self.class_to_idxs[class_][perm]
					self.used_class_idxs_count[class_] = 0
			for _ in range(self.episode_length):
				yield batch
				count += self.batch_size
			episode += 1

	def __len__(self):
		return self.episodes * self.episode_length
