# Construct layers and models
# @author: _john
# =============================================================================
import pdb

import numpy as np
import torch
import torch.nn as nn


def mask_backward_gradient_fc(module, grad_input, grad_output):
    """Perform masking of the gradient
    
    # Arguments
        module [nn.Module]: this module
        grad_input [Tensor]: the input gradient
        grad_output [Tensor]: the output gradient
    """
    grad_biases = grad_input[0] * module._mask_matrix_b
    grad_weights = grad_input[2] * module._mask_matrix_w
    return (grad_biases, grad_input[1], grad_weights)


def mask_backward_gradient_conv(module, grad_input, grad_output):
    """Perform masking of the gradient
    
    # Arguments
        module [nn.Module]: this module
        grad_input [Tensor]: the input gradient
        grad_output [Tensor]: the output gradient
    """
    grad_biases = grad_input[2] * module._mask_matrix_b
    grad_weights = grad_input[1] * module._mask_matrix_w
    return (grad_input[0], grad_weights, grad_biases)


class LinearSubspace(nn.Linear):
    """Implement normal dense layer, except that it will only update a fixed
    set of weights

    @NOTE: this is a specific version of linear projection, in which the
    subspace is constructed by directly dropping some dimensions and retaining
    the others.

    @TODO: create a subspace transformation using projection.

    # Arguments
        subspace [int]: the size of subspace
        in_features, out_features, bias: refer to nn.Linear
    """

    def __init__(self, in_features, out_features, subspace, bias=True):
        """Initialize the layer"""
        super(LinearSubspace, self).__init__(in_features, out_features, bias)

        # constructing the mask matrix
        n_weight_params = out_features * in_features
        n_bias_params = out_features

        mask_matrix = np.zeros(shape=n_weight_params + n_bias_params)
        activated_index = np.random.choice(
            len(mask_matrix), subspace, replace=False)
        mask_matrix[activated_index] = 1
        self._mask_matrix_w = torch.Tensor(
            mask_matrix[:n_weight_params].reshape(in_features,out_features)
        ).cuda()
        self._mask_matrix_b = torch.Tensor(
            mask_matrix[n_weight_params:].reshape(out_features)).cuda()
        self._mask_matrix_w.requires_grad = False
        self._mask_matrix_b.requires_grad = False

        # register the backward hook
        self.mg_handle = self.register_backward_hook(mask_backward_gradient_fc)

    def remove_gradient_mask(self):
        """Remove the gradient masking operation"""
        self.mg_handle.remove()


class Conv2dSubspace(nn.Conv2d):
    """Implement normal convolution 2D layer, except that it will only update
    a fixed set of weights

    @NOTE: this is a specific version of linear projection, in which the
    subspace is constructed by directly dropping some dimensions and retaining
    the others.

    @TODO: create a subspace transformation using projection.

    # Arguments:
        subspace [int]: the number of subspace dimensions
        [...]: refer to `torch.nn.Conv2d`
    """

    def __init__(self, in_channels, out_channels, kernel_size, subspace,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSubspace, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)

        # construct the mask matrix
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int)
            else kernel_size)
        n_weight_params = in_channels * out_channels * np.prod(kernel_size)
        n_bias_params = out_channels
        mask_matrix = np.zeros(shape=n_weight_params+n_bias_params)
        activated_index = np.random.choice(
            len(mask_matrix), subspace, replace=False)
        mask_matrix[activated_index] = 1
        self._mask_matrix_w = torch.Tensor(
            mask_matrix[:n_weight_params]
                .reshape(out_channels, in_channels, *kernel_size)).cuda()
        self._mask_matrix_b = torch.Tensor(
            mask_matrix[n_weight_params:].reshape(out_channels)).cuda()
        self._mask_matrix_w.requires_grad = False
        self._mask_matrix_b.requires_grad = False

        # register the backward hook
        self.mg_handle = self.register_backward_hook(mask_backward_gradient_conv)
    
    def remove_gradient_mask(self):
        """Remove the gradient masking operation"""
        self.mg_handle.remove()
        

