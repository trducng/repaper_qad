"""Pytorch implementation inspired from
http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    """The CPPN generator vector.
    
    This model takes 4 values: x, y, r, z and output a single scalar
    """

    def __init__(self, z_dim):
        """Initialize the object"""
        super(Generator, self).__init__()

        # self.x_dim = x_dim
        # self.y_dim = y_dim
        self.z_dim = z_dim

        self.x_path = nn.Linear(in_features=1, out_features=64, bias=False)
        self.y_path = nn.Linear(in_features=1, out_features=64, bias=False)
        self.r_path = nn.Linear(in_features=1, out_features=64, bias=False)
        self.z_path = nn.Linear(in_features=z_dim, out_features=64)
        
        self.hidden = nn.Linear(in_features=64, out_features=64)
        self.output = nn.Linear(in_features=64, out_features=1)
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.5)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias)

    def forward(self, x, y, r, z):
        """Perform the forward pass

        The arguments x, y and r must be scaled to be withing range 0 and 1

        # Arguments
            x [2D tensor]: B x 1
            y [2D tensor]: B x 1
            r [2D tensor]: B x 1
            z [2D tensor]: B x z_dim
        
        # Returns
            [2D tensor]: B x 1
        """
        hidden_x = self.x_path(x)
        hidden_y = self.y_path(y)
        hidden_r = self.r_path(r)
        hidden_z = self.z_path(z)

        hidden = torch.tanh(hidden_x + hidden_y + hidden_r + hidden_z)

        residual = hidden
        hidden = torch.tanh(self.hidden(hidden))
        hidden = torch.tanh(self.hidden(hidden))
        hidden = torch.tanh(self.hidden(hidden))
        hidden += residual

        return torch.sigmoid(self.output(hidden))
    
    def get_z_dim(self):
        """Return the z dimension"""

        return self.z_dim


def sample_images(model, width, height):
    """Sample images with the supplied model"""
    width, height = int(width), int(height)
    if width <= 1 or height <= 1:
        raise AttributeError('width and height must be integer larger than 1')

    z = torch.randn(1, model.get_z_dim())
    xs = torch.arange(width).float() / (width - 1)
    xs = xs.expand(height, width).transpose(0, 1).flatten().float()
    ys = torch.arange(height).float() / (height - 1)
    ys = ys.expand(width, height).flatten().float()
    rs = torch.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2)

    xs, ys, rs = xs.unsqueeze(1), ys.unsqueeze(1), rs.unsqueeze(1)
    zs = z.expand(xs.size(0), model.get_z_dim())
    output = model(xs, ys, rs, zs)
    output = output.squeeze().cpu()

    # from this x, y, r setup, the image is generated from bottom left: go up
    # then go right
    image = output.view(width, height)
    image = image.transpose(0, 1)
    image = image.flip([0])

    return image.data.numpy()

def main():

    model = Generator(32)
    while True:
        image = sample_images(model, 1024, 1024)
        plt.imshow(image, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
