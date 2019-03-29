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


def init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        nn.init.zeros_(module.bias)


class Discriminator(nn.Module):
    """GAN discriminator"""

    def __init__(self):
        """Initialize the discriminator"""
        super(Discriminator, self).__init__()

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=3, stride=2, padding=1),      # 14 x 14
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv_y = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8,
                      kernel_size=3, stride=2, padding=1),      # 14 x 14
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv_layers = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=2, padding=1),      # 7 x 7
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=2, padding=1),      # 4 x 4
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
        )
        self.apply(init)

    def forward(self, x, y):
        """Perform the forward pass"""
        hidden_x = self.conv_x(x)
        hidden_y = self.conv_y(y)

        hidden = torch.cat([hidden_x, hidden_y], dim=1)
        hidden = self.conv_layers(hidden)

        return hidden.view(x.size(0), 1)


class Generator(nn.Module):

    def __init__(self):
        """Initialize the generator"""
        super(Generator, self).__init__()

        self.conv_z = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=128,
                               kernel_size=4, stride=1),                # 4 x 4
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv_y = nn.Sequential(
            nn.ConvTranspose2d(in_channels=10, out_channels=128,
                               kernel_size=4, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv_layers = nn.Sequential(

            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, stride=1),                # 7 x 7
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4),   # 10 x 10
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=4, stride=2),    # 22 x 22
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=7),      # 28 x 28
            nn.Tanh()
        )

        self.apply(init)

    def forward(self, z, y):
        """Perform the forward pass

        # Arguments
            z [4D tensor]: shape [B, 100, 1, 1]
            y [4D tensor]: shape [B, 10, 1, 1]

        # Returns
            [4D tensor]: shape [B, 1, 28, 28]
        """
        hidden_z = self.conv_z(z)
        hidden_y = self.conv_y(y)
        hidden = torch.cat([hidden_z, hidden_y], dim=1)
        return self.conv_layers(hidden)



def main(folder_name):
    train_loader, test_loader = get_dataset()
    dis = Discriminator().cuda()
    gen = Generator().cuda()

    dis_optim = optim.Adam(params=dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
    gen_optim = optim.Adam(params=gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(25, 100, 1, 1).cuda()
    fixed_y = torch.zeros(25, 10, 1, 1).float()
    fixed_y_label = torch.LongTensor(25).random_(0, 10)
    fixed_y[torch.arange(25), fixed_y_label, :, :] = 1.0
    fixed_y = fixed_y.cuda()
    fixed_y_label = list(fixed_y_label.squeeze().cpu().data.numpy())

    epochs = 20

    for epoch in range(epochs):
        for idx, (x, y_) in enumerate(train_loader):
            batch_size = y_.size(0)

            # x = x.view(batch_size, -1).cuda().contiguous()
            x = x.cuda()
            y_gen = torch.zeros(batch_size, 10, 1, 1).float()
            y_gen[torch.arange(batch_size), y_, :, :] = 1.0
            y_gen = y_gen.cuda().contiguous()
            y_dis = torch.zeros(batch_size, 10, 28, 28).float()
            y_dis[torch.arange(batch_size), y_, :, :] = 1.0
            y_dis = y_dis.cuda().contiguous()

            labels_real = torch.ones(batch_size, 1).cuda()
            labels_fake = torch.zeros(batch_size, 1).cuda()

            # if idx % 200 == 0:
            #     pdb.set_trace()

            # Discriminator optim
            output_real = dis(x, y_dis)
            dis_loss_real = criterion(output_real, labels_real)

            z = torch.randn(batch_size, 100, 1, 1).cuda()
            x_fake = gen(z, y_gen)
            output_fake = dis(x_fake, y_dis)
            dis_loss_fake = criterion(output_fake, labels_fake)

            dis_loss = dis_loss_real + dis_loss_fake
            dis_optim.zero_grad()
            gen_optim.zero_grad()
            dis_loss.backward()
            dis_optim.step()

            # Generator optim
            z = torch.randn(batch_size, 100, 1, 1).cuda()
            y_fake = torch.LongTensor(batch_size).random_(0, 10)
            y_gen = torch.zeros(batch_size, 10, 1, 1).float()
            y_gen[torch.arange(batch_size), y_fake, :, :] = 1.0
            y_gen = y_gen.cuda().contiguous()
            y_dis = torch.zeros(batch_size, 10, 28, 28).float()
            y_dis[torch.arange(batch_size), y_fake, :, :] = 1.0
            y_dis = y_dis.cuda().contiguous()

            x_fake = gen(z, y_gen)
            output_fake = dis(x_fake, y_dis)
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
            ax.title.set_fontsize(10)
            ax.imshow(each_image, cmap='gray')
            ax.axis('off')

        plt.tight_layout(pad=0.5)
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
