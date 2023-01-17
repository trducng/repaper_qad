import torch


class NonLinear(torch.nn.Module):
	def __init__(self, in_chns, hidden_chns, num_classes):
		super().__init__()
		self.in_chns = in_chns
		self.hidden_chns = hidden_chns
		self.num_classes = num_classes

		self.classify = torch.nn.Sequential(
			torch.nn.Linear(in_chns, hidden_chns),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_chns, num_classes)
		)

	def forward(self, x):
		return self.classify(x)
