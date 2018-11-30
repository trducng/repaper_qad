# Implement osme models to explore basic intrinsic
# @author: _john
# ============================================================================
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LinearSubspace, Conv2dSubspace


class FullyConnectedSubspace(nn.Module):
    """Create a fully connected layer with random degree of freedom
    
    # Arguments
        in_features [int]: the number of input features
        out_features [int]: the number of output features
        subspace_dim [int]: the dimension of free subspace
    """

    def __init__(self, in_features, out_features, subspace_dim):
        """Initialize the model"""
        super(FullyConnectedSubspace, self).__init__()

        # get random intrinsic dim of each dimension
        free = np.random.choice(subspace_dim, 3, replace=True)
        free = (free / np.sum(free) * subspace_dim).astype(np.uint64)
        if np.sum(free) != subspace_dim:
            # can have slight misfits, due to changing from float to integer
            free[np.random.choice(len(free), 1)[0]] += (subspace_dim - np.sum(free))

        self.hidden1 = LinearSubspace(in_features, 200, free[0])
        self.hidden2 = LinearSubspace(200, 200, free[1])
        self.output = LinearSubspace(200, out_features, free[2])

    def forward(self, x):
        """Perform the forward pass"""
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        y = self.output(x)

        return y
    
    def remove_gradient_mask(self):
        """Remove all the gradient masking"""
        for each_module in self.children():
            if hasattr(each_module, 'remove_gradient_mask'):
                each_module.remove_gradient_mask()

class LeNetSubspace(nn.Module):
    """Create a LeNet with random degree of freedom

    # Arguments
        in_channels [int]: the number of input channels
        out_features [int]: the number of output features
        subspace_dim [int]: the dimension of free subspace
    """

    def __init__(self, in_channels, out_features, subspace_dim):
        """Initialize the object"""
        super(LeNetSubspace, self).__init__()
        
        # get random intrinsic dim of each dimension
        free = np.random.choice(subspace_dim, 4, replace=True)
        free = (free / np.sum(free) * subspace_dim).astype(np.uint64)
        if np.sum(free) != subspace_dim:
            # can have slight misfits, due to changing from float to integer
            free[np.random.choice(len(free), 1)[0]] += (subspace_dim - np.sum(free))

        self.conv1 = Conv2dSubspace(
            in_channels=in_channels, out_channels=20, subspace=free[0],
            kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = Conv2dSubspace(
            in_channels=20, out_channels=50, subspace=free[1],
            kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.full_hidden = LinearSubspace(
            in_features=7*7*50, out_features=500, subspace=free[2])
        self.output = LinearSubspace(in_features=500, out_features=10,
            subspace=free[3])

    def forward(self, x):
        """Perform the forward pass"""
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.full_hidden(x))
        
        return self.output(x)

    def remove_gradient_mask(self):
        """Remove all the gradient masking"""
        for each_module in self.children():
            if hasattr(each_module, 'remove_gradient_mask'):
                each_module.remove_gradient_mask()
