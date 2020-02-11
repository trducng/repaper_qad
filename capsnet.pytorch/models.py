"""Reimplement CapsuleNetwork, based on *Dynamic Routing Between Capsules*"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


def caps_squash(input_x):
    """Squashing activation in Capsule Unit

    # Arguments
        input_x [2D tensor]: batch x n_channels
    """
    norm = torch.norm(input_x, p=2, dim=1)
    return input_x * (norm ** 2) / ((1 + norm ** 2) * norm)


def squash(input_x):
    """Squashing activation in Capsule layer

    # Args
        input_x [5D tensor]: b x n_units x c
    """
    norm = torch.norm(input_x, p=2, dim=2, keepdim=True)
    output = input_x * (norm ** 2) / ((1 + norm ** 2) * norm)
    return output


class CapsuleUnit(nn.Module):
    """The capsule unit inside capsule layer"""

    def __init__(self, in_channels, out_channels, in_caps):
        super(CapsuleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_caps = in_caps
        self.n_routing_iterations = 10

        # this might be convolution operation
        self.W = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.coeffs = torch.ones([1, in_caps, 1], dtype=torch.float32)
        self.coeffs.detach()

    def forward(self, input_x):
        """Perform forward pass

        # Args
            input_x [3D tensors]: b x n_capsules x in_channels
        """
        u_ji = self.W(input_x)  # b * n_capsules * out_channels
        for each_iteration in self.n_routing_iterations:
            c_ij = torch.softmax(self.coeffs, dim=1)  # 1 * n_capsules * 1
            sj = u_ji * c_ij  # b * n_capsules * out_channels
            sj = sj.sum(dim=1, keepdim=True)  # b * 1 * out_channels
            vj = caps_squash(sj).squeeze(1)  # b * out_channels

            update = torch.bmm(u_ji, vj.T)  # b * n_capsules * 1
            self.coeffs += update.sum(0, keepdim=True)  # 1 * n_capsules * 1

        return vj


class CapsuleLayer(nn.Module):
    """The capsule layer"""

    def __init__(self, in_units, out_units, in_channels, out_channels, n_iterations):
        super(CapsuleLayer, self).__init__()
        self.in_units = in_units
        self.out_units = out_units
        self.out_channels = out_channels
        self.n_iterations = n_iterations

        # self.W = nn.Parameter(torch.Tensor(1, out_units, in_channels, out_channels))
        self.W = nn.Parameter(
            torch.randn(in_units, in_channels, out_channels * out_units)
        )

    def forward(self, input_x):
        """Perform the forward pass

        # Args
            input_x [3D tensor]: b * in_units * in_channels
        """
        # we want to transform into
        # b x out_units x in_units x out_channels or equivalent
        # then we will do weighted average along the in_units dimension to get
        # b x out_units * out_channels
        hidden = input_x.unsqueeze(2)  # b * in_units * 1 * in_channels
        hidden = torch.matmul(hidden, self.W)  # b * in_units * 1 * out
        hidden = hidden.squeeze(2)  # b * in_units * out
        u_ji = hidden.view(
            hidden.size(0), hidden.size(1), self.out_units, self.out_channels
        )  # b * in_units * ounit * ochans

        b = torch.zeros(
            [u_ji.size(0), self.in_units, self.out_units], dtype=torch.float32
        )
        if input_x.is_cuda:
            b = b.cuda()
        b = b.detach()
        for each_iteration in range(self.n_iterations):
            c_i = torch.softmax(b, dim=1)  # b * in_units * out_units
            c_i = c_i.unsqueeze(-1)  # b * in_units * out_units * 1
            s_j = c_i * u_ji  # b * in_units * ounits * ochans
            s_j = s_j.sum(1)  # b * ounits * ochans
            v_j = squash(s_j)  # b * ounits * ochans

            with torch.no_grad():
                u_ji_temp = u_ji.permute(0, 2, 1, 3)  # b * ounit * in_units * ochans
                u_ji_temp = u_ji_temp.reshape(
                    -1, self.in_units, self.out_channels
                )  # comp * in_units * oc
                v_j_temp = v_j.view(-1, self.out_channels).unsqueeze(
                    -1
                )  # comp * oc * 1
                updates = torch.bmm(u_ji_temp, v_j_temp)  # comp * in_units * 1
                updates = updates.view(-1, self.out_units, self.in_units).permute(
                    0, 2, 1
                )  # b * in_units * out_units

                b = b + updates

                # another way to get the updates
                # v_j_temp = v_j.unsqueeze(1)  # b * 1 * ounits * ochans
                # updates = (v_j_temp * u_ji).sum(-1)

        return v_j


class PrimaryCapsuleLayer(nn.Module):
    """Primary layer of the CapsuleNetwork

    In the paper, the second layer (PrimaryCapsules) is a convolutional capsule layer
    with:
        - 32 channels of 8D capsules
        - each 8D capsule contains 8 convolutional unit: 9x9 kernel, stride 2
        - each capsule output sees the outputs of all CxHxW earlier units whoose
            receptive fields overlap with the location of the  center of the capsule
    """

    def __init__(self, in_channels, out_channels, n_units, kernel_size, stride=1):
        super(PrimaryCapsuleLayer, self).__init__()
        self.out_channels = out_channels
        self.n_units = n_units

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * n_units,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, input_x):
        """Perform the forward pass

        # Args
            input_x [4D tensor]: b x c x h x w

        # Returns
            [3D tensor]: b x [n_units * hr * wr] x out_channels
        """
        hidden = self.conv(input_x)
        b, c, h, w = hidden.shape

        hidden = hidden.reshape(b, self.n_units, self.out_channels, h, w)
        hidden = hidden.permute(0, 1, 3, 4, 2)  # b * u * h * w * c
        hidden = hidden.reshape(b, -1, self.out_channels)
        output = squash(hidden)

        return output


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


if __name__ == "__main__":
    cuda = True
    epoch = 50
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            "datasets", download=True, transform=torchvision.transforms.ToTensor()
        ),
        batch_size=128,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
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
        optimizer, mode='max', patience=2, verbose=True)
    for each_epoch in range(epoch):
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
                    images, f'logs/output_{each_epoch}_{idx}.png', nrow=5)

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
                    images, f'logs/test_{each_epoch}_{idx}.png', nrow=10)

        print(f"Test accuracy: {corrects/len(test_loader.dataset)}")
        lr_scheduler.step(corrects / len(test_loader.dataset))
        torch.save({
            'capsnet': model.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': each_epoch
        }, f'logs/capsnet_{each_epoch}.pth')
