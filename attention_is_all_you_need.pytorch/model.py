import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SelfAttention(nn.Module):
    """An encoder layer

    # Arguments
        input_shape [int]: the size of each input vector
        qk_shape [int]: the size of query and key vectors
        v_shape [int]: the output vector size
        n_heads [int]: the number of attention heads
    """

    def __init__(self, input_shape, qk_shape, v_shape):
        """Initialize the object"""
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(in_features=input_shape, out_features=qk_shape)
        self.keys = nn.Linear(in_features=input_shape, out_features=qk_shape)
        self.values = nn.Linear(in_features=input_shape, out_features=v_shape)

    def forward(self, input_, mask=None):
        """Perform the forward pass

        # Arguments
            input_ [Timestep x In_Channels]: 2D input
        
        # Returns
            [Timestep x Out_Channels]: 2D output
        """
        _query = self.query(input_)         # [Timestep x qk_shape]
        _keys = self.keys(input_)           # [Timestep x qk_shape]
        _values = self.values(input_)       # [Timestep x v_shape]

        attention = torch.matmul(_query, _keys.transpose(0, 1))     # [Timestep x Timestep]
        attention = attention / math.sqrt(attention.numel())

        if mask is not None:
            attention = attention * mask

        attention = torch.softmax(attention, dim=1)
        output = torch.matmul(attention, _values)                   # [Timestep x v_shape]

        # @NOTE on `output`:
        # _values has the shape of [Timestep x v_shape], each element in the
        # timestep is the embedded vector of v_shape. `attention` has the shape
        # of [Timestep x Timestep], each timestep is a vector of Timestep shape,
        # each element in that vector is how much we should attend to the other
        # element (e.g, the first element is how much we attend to the first
        # timesstep, the second element is how much we attend to the second
        # timestep...). Output has the shape of [Timestep x v_shape], each
        # timestep is the element averaged by the attended original values.
        return output


class FeedForward(nn.Module):
    """Feed-forward inside each encoder and decoder"""
    def __init__(self, input_shape, hidden_shape):
        """Initialize the object"""
        self.linear1 = nn.Linear(in_features=input_shape, out_features=hidden_shape)
        self.linear2 = nn.Linear(in_features=hidden_shape, out_features=input_shape)
    
    def forward(self, input_):
        """Perform the forward pass"""
        hidden = torch.relu(self.linear1(input_))
        output = self.linear2(hidden)
        return output


class PositionalEncoding(nn.Module):
    """Perform positional encoding"""
    
    def forward(self, input_):
        """Perform the forward pass
        
        # Arguments
            input_ [Timestep x Dimension]: the embedded input
        """
        timestep, dim = input_.shape
        position = torch.arange(timestep).unsqueeze(1)          # Timestep x 1
        pe = torch.zeros(timestep, dim)                         # Timestep x Dimension
        div_term = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim)).unsqueeze(0)
        # 1 x Dimension / 2
        pe[:, 0::2] = torch.sin(torch.matmul(position, div_term))  # Timestep x Dimension / 2
        pe[:, 1::2] = torch.cos(torch.matmul(position, div_term))  # Timestep x Dimension / 2

        output = pe + input_
        return output


class SubLayer(nn.Module):
    """Each sublayer wraps the module, add residual connection and layer
    normalization"""

    def __init__(self, input_shape, output_shape, module):
        """Initialize the object"""
        super(SubLayer, self).__init__()

        self.linear = None
        if input_shape != output_shape:
            self.linear = nn.Linear(in_features=input_shape, out_features=output_shape)
        self.layer_norm = nn.LayerNorm(normalized_shape=output_shape)
        self.module = module

    def forward(self, input_):
        """Perform the forward pass"""
        hidden = self.module(input_)
        if self.linear is not None:
            residual = self.linear(input_)
            hidden += residual

        hidden = self.layer_norm(hidden)
        return hidden


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            SubLayer(512, 512, SelfAttention(512, 512, 512)),
            SubLayer(512, 512, FeedForward(512, 2048)),

            SubLayer(512, 512, SelfAttention(512, 512, 512)),
            SubLayer(512, 512, FeedForward(512, 2048)),

            SubLayer(512, 512, SelfAttention(512, 512, 512)),
            SubLayer(512, 512, FeedForward(512, 2048)),

            SubLayer(512, 512, SelfAttention(512, 512, 512)),
            SubLayer(512, 512, FeedForward(512, 2048)),

            SubLayer(512, 512, SelfAttention(512, 512, 512)),
            SubLayer(512, 512, FeedForward(512, 2048)),

            SubLayer(512, 512, SelfAttention(512, 512, 512)),
            SubLayer(512, 512, FeedForward(512, 2048)),
        )


if __name__ == '__main__':
    layer = SelfAttention(input_shape=10, qk_shape=10, v_shape=15)
    input_ = torch.randn(5, 10)
    output_ = layer(input_)
    print(output_.shape)