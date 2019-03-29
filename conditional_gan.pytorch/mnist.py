"""Quick and dirty conditional GAN implementation"""
import os
import pdb

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])
BATCH_SIZE = 128



class MaxOut(nn.Module):

    def __init__(self, in_features, out_features, pieces):
        """Initialize the object"""
        super(MaxOut, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pieces = pieces
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features * pieces)

    def forward(self, input_):
        """Perform the forward pass"""
        hidden = self.linear(input_)
        hidden = hidden.view(input_.size(0), self.out_features, self.pieces)
        return hidden.max(-1)[0]


def get_dataset():
    train_data = datasets.MNIST(
        'datasets', train=True, download=True, transform=TRANSFORM)
    test_data = datasets.MNIST(
        'datasets', train=False, download=True, transform=TRANSFORM)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE)

    return train_loader, test_loader


class Discriminator(nn.Module):
    """The conditional GAN discriminator for MNIST"""

    def __init__(self):
        """Initialize the object"""
        super(Discriminator, self).__init__()
        self.max_x = MaxOut(in_features=784, out_features=240, pieces=5)
        self.max_y = MaxOut(in_features=10, out_features=50, pieces=5)
        self.max_hidden = MaxOut(in_features=290, out_features=240, pieces=5)
        self.output = nn.Linear(in_features=240, out_features=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, y):
        """Perform the forward pass

        # Arguments
            x [2D tensor]: shape B x 784
            y [2D tensor]: shape B x 10

        # Returns
            0D tensor: just a number
        """
        hidden_x = self.dropout(self.max_x(x))
        hidden_y = self.dropout(self.max_y(y))
        hidden = self.max_hidden(torch.cat([hidden_x, hidden_y], dim=-1))
        hidden = self.dropout(hidden)
        return self.output(hidden)


class Generator(nn.Module):
    """The GAN generator for MNIST"""
    
    def __init__(self):
        """Initialize the object"""
        super(Generator, self).__init__()
        self.hidden_z = nn.Linear(in_features=100, out_features=200)
        self.hidden_y = nn.Linear(in_features=10, out_features=1000)
        self.hidden = nn.Linear(in_features=1200, out_features=1200)
        self.output = nn.Linear(in_features=1200, out_features=784)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, z, y):
        """Perform the forward pass
        
        # Arguments
            z [2D tensor]: shape B x 100
            y [2D tensor]: shape B x 10
        
        # Returns
            [2D tensor]: shape B x 784
        """
        hidden_z = self.dropout(torch.relu(self.hidden_z(z)))
        hidden_y = self.dropout(torch.relu(self.hidden_y(y)))
        hidden = torch.relu(torch.cat([hidden_z, hidden_y], dim=-1))
        hidden = self.dropout(hidden)
        return F.tanh(self.output(hidden))


def main(folder_name):
    train_loader, test_loader = get_dataset()
    dis = Discriminator().cuda()
    gen = Generator().cuda()

    dis_optim = optim.Adam(params=dis.parameters(), lr=1e-4, betas=(0.5, 0.999))
    gen_optim = optim.Adam(params=gen.parameters(), lr=1e-3, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(25, 100).cuda()
    fixed_y = torch.zeros(25, 10).float()
    fixed_y_label = torch.LongTensor(25).random_(0, 10)
    fixed_y[torch.arange(25), fixed_y_label] = 1.0
    fixed_y = fixed_y.cuda()
    fixed_y_label = list(fixed_y_label.squeeze().cpu().data.numpy())

    epochs = 20

    for epoch in range(epochs):
        for idx, (x, y_) in enumerate(train_loader):
            batch_size = y_.size(0)

            x = x.view(batch_size, -1).cuda().contiguous()
            y = torch.zeros(batch_size, 10).float()
            y[torch.arange(batch_size), y_] = 1.0
            y = y.cuda().contiguous()

            labels_real = torch.ones(batch_size, 1).cuda()
            labels_fake = torch.zeros(batch_size, 1).cuda()

            # if idx % 200 == 0:
            #     pdb.set_trace()

            # Discriminator optim
            output_real = dis(x, y)
            dis_loss_real = criterion(output_real, labels_real)

            z = torch.randn(batch_size, 100).cuda()
            x_fake = gen(z, y)
            output_fake = dis(x_fake, y)
            dis_loss_fake = criterion(output_fake, labels_fake)

            dis_loss = dis_loss_real + dis_loss_fake
            dis_optim.zero_grad()
            gen_optim.zero_grad()
            dis_loss.backward()
            dis_optim.step()

            # Generator optim
            z = torch.randn(batch_size, 100).cuda()
            x_fake = gen(z, y)
            output_fake = dis(x_fake, y)
            gen_loss = criterion(output_fake, labels_real)

            dis_optim.zero_grad()
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            if idx % 200 == 0:
                print('[{}|{}]: dis loss {} - gen loss {}'.format(
                    epoch, idx, dis_loss.item(), gen_loss.item()
                ))

        gen.eval()
        x_fake = gen(fixed_z, fixed_y)
        gen.train()

        images = x_fake.view(25, 28, 28).cpu().data.numpy()
        epoch_images = []
        for image_index in range(25):
            image = images[image_index]
            image = image - np.min(image)
            image = image / np.max(image)
            image = (image * 255).astype(np.uint8)
            epoch_images.append(image)

        plt.figure()
        for _idx, each_image in enumerate(epoch_images):
            ax = plt.subplot(5, 5, _idx + 1)
            ax.set_title('{}'.format(fixed_y_label[_idx]))
            ax.imshow(each_image, cmap='gray')
            ax.axis('off')

        plt.suptitle('Epoch {}'.format(epoch))
        plt.savefig(os.path.join(folder_name, '{}.png'.format(epoch)))
        plt.close()

    images = []
    for epoch in range(epochs):
        image_name = os.path.join(folder_name, '{}.png'.format(epoch))
        images.append(imageio.imread(image_name))
    imageio.mimsave(os.path.join(folder_name, 'final.gif'), images, fps=5)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()

    folder_name = os.path.join('images', args.name)
    if os.path.exists(folder_name):
        import shutil
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)
    main(folder_name)
