import torch


class Linear(torch.nn.Module):
	def __init__(self, in_chns, num_classes):
		super().__init__()
		self.in_chns = in_chns
		self.num_classes = num_classes
		self.classify = torch.nn.Linear(self.in_chns, self.num_classes)

	def forward(self, x):
		return self.classify(x)
