"""Reimplement CapsuleNetwork, based on *Dynamic Routing Between Capsules*"""
import math
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasource import download
from datasource.loaders import smallnorb_loader


class CapsuleLayer(nn.Module):
    """The capsule layer"""

    def __init__(self, in_units, out_units, m_size, lamb, lamb_mult, n_iterations):
        super(CapsuleLayer, self).__init__()
        self.in_units = in_units
        self.out_units = out_units
        self.m_size = m_size
        self.out_channels = m_size ** 2
        self.n_iterations = n_iterations

        self.W = nn.Parameter(
            torch.randn(in_units, out_units, m_size, m_size)
        )
        self.beta_u = nn.Parameter(torch.randn(out_units))
        self.beta_a = nn.Parameter(torch.randn(out_units))
        self.lamb = lamb
        self.lamb_mult = lamb_mult

    def forward(self, input_a, input_M):
        """Perform the forward pass

        # Args
            input_a [2D tensor]: probability (b * in_units)
            input_M [4D tensor]: pose matrices (b * in_units * m_size * m_size)

        # Returns
            [2D tensor]: probability (b * out_units)
            [3D tensor]: pose matrices (b * out_units * m_size * m_size)
        """
        # we will need to get the vote matrix of shape
        # b * out_units * in_units * m_size * m_size
        # so that later on, we can get sum along the in_units dimension to obtain
        # pose matrix for this layer: b * out_units * m_size * m_size
        b = input_M.size(0)
        hidden = input_M.unsqueeze(2)  # b * in_units * 1 * m_size * m_size
        v = torch.matmul(hidden, self.W)  # b * in_units * out_unit * msize * msize
        v_reshape = v.reshape(b, self.in_units, self.out_units, self.out_channels)
        v_reshape = v.permute(0, 2, 3, 1) # b * channels * out_units * in_units

        self.R = torch.ones(
            (hidden.size(0), out_units, in_units), dtype=torch.float32) / out_units
        beta_u = self.beta_u.expand(b, self.out_units)  # b * out_units
        beta_a = self.beta_a.expand(b, self.out_units)  # b * out_units
        if input_a.is_cuda:
            self.R = self.R.cuda()
        for each_iteration in range(self.n_iterations):
            # maximization step
            self.R = self.R * input_a.unsqueeze(1)     # b * out_units * in_units
            self.R = self.R.unqueeze(1) # b * 1 * out_units * in_units
            sum_R = self.R.sum(dim=3)   # b * 1 * out_units

            mu = self.R * v_reshape # b * channels * out_units * in_units
            mu = mu.sum(dim=3) # b * channels * out_units
            mu = mu / sum_R # b * channels * out_units

            var = self.R * (v_reshape - mu.unsqueeze(-1)).pow(2)
            var = var.sum(dim=3)
            var = var / sum_R
            sigma = var.sqrt() # b * channels * out_units

            cost = (beta_u.unsqueeze(1) + np.log(sigma)) * sum_R # b * channels * out_units
            lamb = self.lamb * self.lamb_mult ** each_iteration
            a = torch.sigmoid(lamb * (beta_a - cost.sum(dim=1)))    # b * out_units

            # expectation step
            exponent = (v_reshape - mu.unsqueeze(-1)).pow(2) / (2 * var.unsqueeze(-1)) # b * channels * out_units * in_units
            exponent = -exponent.sum(dim=1) # b * out_units * in_units
            eff = (2 * math.pi * var.unsqueeze(-1)).prod(dim=1) # b * out_units * 1
            p = eff * torch.exp(exponent)   # b * out_units * in_units

            numerator = a.unsqueeze(-1) * p # b * out_units * in_units
            self.R = numerator / numerator.sum(dim=1, keepdim=True) # b * out_units * in_units

        return a, mu


class PrimaryCapsuleLayer(nn.Module):
    """Primary layer of the CapsuleNetwork

    In the paper, the second layer (PrimaryCapsules) is a convolutional capsule layer
    with:
        - 32 channels of 8D capsules
        - each 8D capsule contains 8 units: 9x9 kernel, stride 2
        - each capsule output sees the outputs of all CxHxW earlier units whoose
            receptive fields overlap with the location of the  center of the capsule
    """

    def __init__(self, in_channels, n_units, m_size, kernel_size, stride):
        super(PrimaryCapsuleLayer, self).__init__()
        self.m_size = m_size
        self.out_channels = m_size * m_size + 1
        self.n_units = n_units

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels * n_units,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, input_x):
        """Perform the forward pass

        # Args
            input_x [4D tensor]: b x c x h x w

        # Returns
            [2D tensor]: b * [n_units * hr * wr]
            [3D tensor]: b * [n_units * hr * wr] x 16
        """
        hidden = self.conv(input_x)
        b, c, h, w = hidden.shape

        hidden = hidden.reshape(b, self.n_units, self.out_channel, h, w)
        hidden = hidden.permute(0, 1, 3, 4, 2)  # b * u * h * w * c
        hidden = hidden.reshape(b, -1, self.out_channels)

        a = hidden[:,:,0]
        M = hidden[:,:,1:].reshape(b, -1, self.m_size, self.m_size).contiguous()

        return torch.sigmoid(a), M


class CapsuleNetwork(nn.Module):
    """The CapsuleNetwork used in "Dynamic Routing between Capsules" """

    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9), nn.ReLU()
        )
        self.layer2 = PrimaryCapsuleLayer(
            in_channels=256, out_channels=8, n_units=32, kernel_size=9, stride=2
        )
        self.layer3 = CapsuleLayer(
            in_units=1152, out_units=10, in_channels=8, out_channels=16, n_iterations=3
        )

    def forward(self, input_x):
        """Perform the forward pass

        # Args
            input_x [4D tensor]: b x c x h x w

        # Returns
            [3D tensor]: b x out_units x out_channels
        """
        hidden = self.layer1(input_x)
        hidden = self.layer2(hidden)
        output = self.layer3(hidden)
        return output


class ReconstructionNetwork(nn.Module):
    """Get reconstruction loss as regularization term"""

    def __init__(self, in_units, in_channels):
        super(ReconstructionNetwork, self).__init__()
        self.in_units = in_units
        self.in_channels = in_channels

        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_units * in_channels, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=784),
            nn.Sigmoid()
        )

    def forward(self, input_x, labels):
        """Perform the forward pass

        # Args
            input_x [3D tensor]: b * n_units * n_channels
            label [1D tensor]: b
        """
        # construct the mask
        mask = torch.zeros((input_x.size(0), self.in_units),
                dtype=torch.float32)
        if input_x.is_cuda:
            mask = mask.cuda()
        mask = mask.detach()
        mask.scatter_(1, labels.view(-1, 1), 1.0)
        mask = mask.unsqueeze(-1)

        # reconstruct image
        input_x = (input_x * mask).contiguous()
        input_x = input_x.view(input_x.size(0), -1)
        output = self.decoder(input_x)
        output = output.reshape(input_x.size(0), 28, 28).unsqueeze(1).contiguous()
        return output


class MarginLoss(nn.Module):
    """Perform margin loss"""

    def __init__(self, lamb=0.5, m_pos=0.9, m_neg=0.1, reduction="mean"):
        super(MarginLoss, self).__init__()
        self.lamb = lamb
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.reduction = reduction

    def forward(self, input_x, target):
        """Perform batch loss calculation

        # Args
            input_x [3D tensor]: b x out_units x out_channels
            target [1D tensor]: b

        # Returns
            [1D tensor]: the aggregated loss
        """
        prob = torch.norm(input_x, p=2, dim=2)  # b * out_units

        mtarget = torch.zeros(prob.size(), dtype=torch.float32)
        if input_x.is_cuda:
            mtarget = mtarget.cuda()
        mtarget.scatter_(1, target.view(-1, 1), 1)

        loss = mtarget * F.relu(self.m_pos - prob).pow(2)
        loss += self.lamb * (1 - mtarget) * F.relu(prob - self.m_neg).pow(2)

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def train(n_epochs=50, log_dir='logs', cuda=True):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.KMNIST(
            "datasets", download=True, transform=torchvision.transforms.ToTensor()
        ),
        batch_size=128,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "datasets",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=128,
        shuffle=True,
    )

    # test forward and backward modele without error
    model = CapsuleNetwork()
    decoder = ReconstructionNetwork(in_units=10, in_channels=16)
    if cuda:
        model = model.cuda()
        decoder = decoder.cuda()
    margin_criterion = MarginLoss()
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, verbose=True)
    for each_epoch in range(n_epochs):
        count, total_loss = 0, 0
        for idx, (x, y) in enumerate(train_loader):
            if cuda:
                x, y = x.cuda(), y.cuda()

            output = model(x)
            recons = decoder(output, y)
            margin_loss = margin_criterion(output, y)
            recons_loss = mse_criterion(recons, x)
            loss = margin_loss + 5e-4 * recons_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            total_loss += loss.item()
            if idx % 100 == 0:
                print(f"IDX {each_epoch} - {idx}: {total_loss / count}, {margin_loss.item()}, {recons_loss.item()}")
                images = torch.cat([x[:5], recons[:5]], dim=0)
                torchvision.utils.save_image(
                    images, Path(log_dir, f'output_{each_epoch}_{idx}.png'), nrow=5)

        # perform the test
        corrects = 0
        model = model.eval()
        for idx, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                if cuda:
                    x, y = x.cuda(), y.cuda()

                output = model(x)
                probs = torch.norm(output, p=2, dim=2)
                preds = torch.argmax(probs, dim=1)
                corrects += (preds == y).sum().item()

            if idx % 80 == 0:
                recons = decoder(output, preds)
                images = torch.cat([x[:10], recons[:10]], dim=0)
                torchvision.utils.save_image(
                    images, Path(log_dir, f'test_{each_epoch}_{idx}.png'), nrow=10)

        print(f"Test accuracy: {corrects/len(test_loader.dataset)}")
        lr_scheduler.step(corrects / len(test_loader.dataset))
        torch.save({
            'capsnet': model.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': each_epoch
        }, Path(log_dir, f'capsnet_{each_epoch}.pth'))

if __name__ == "__main__":
    # fire.Fire(train)
    folder_name = 'datasets'
    download('smallNORB', folder_name, tags='train')
    x, y = smallnorb_loader(folder_name, train=True)
    print(x.shape, y.shape)
