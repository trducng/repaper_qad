"""Test on MNIST

- TODO:
    . batch norm for generator also
    . increase batch size to 128
    . normalize the image to mean 0.5, std 0.5
    . increase learning rate to 2e-4
    . leakyrelu of negative slope 0.2
    . remove fully-connected layers
    . replace pooling with stride convolution
    . tailor the generator's input: the discriminator converges fast, gen's lost increases,
        however seems to deal the most effectively with mode collapse
    . Adam's betas1 = 0.5: seems better
    . weight zero mean, 0.02 std normal distribution: the final generated images seem clearer
    . copy the training procedure (seems like no problem)
    + copy discriminator architecture
    + copy generator architecture
        . smoother gen
        . smaller learning rate
"""

from PIL import Image
import os
import imageio
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])
BATCH_SIZE = 128


def get_dataset():
    train_data = torchvision.datasets.MNIST(
        'datasets', train=True, download=True, transform=TRANSFORM)
    test_data = torchvision.datasets.MNIST(
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


class DiscriminatorMNIST(nn.Module):
    """GAN discriminator"""

    def __init__(self):
        """Initialize the discriminator"""
        super(DiscriminatorMNIST, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=2, padding=1),      # 14 x 14
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=2, padding=1),      # 7 x 7
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=1),      # 4 x 4
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=128, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
        )
        self.apply(init)

    def forward(self, input_x):
        """Perform the forward pass"""
        hidden = self.conv_layers(input_x)

        return hidden.view(input_x.size(0), 1)


class GeneratorMNIST(nn.Module):

    def __init__(self):
        """Initialize the generator"""
        super(GeneratorMNIST, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=256,
                               kernel_size=4, stride=1),                # 4 x 4
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),

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

    def forward(self, input_x):
        """Perform the forward pass

        # Arguments
            input_x [4D tensor]: shape [B, 512, 8, 8]

        # Returns
            [4D tensor]: shape [B, 1, 28, 28]
        """

        return self.conv_layers(input_x)


def main(folder_name):
    train_loader, test_loader = get_dataset()
    dis = DiscriminatorMNIST().cuda()
    gen = GeneratorMNIST().cuda()

    fixed_z = torch.randn(25, 100, 1, 1).cuda()
    progress_images = []

    dis_optimizer = optim.Adam(params=dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
    gen_optimizer = optim.Adam(params=gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    epochs = 20

    for epoch in range(epochs):
        for idx, (X, _) in enumerate(train_loader):
            X = X.cuda().contiguous()
            batch_size = X.size(0)
            labels_real = torch.ones(batch_size, 1).cuda()
            labels_fake = torch.zeros(batch_size, 1).cuda()

            # Discriminator
            output_real = dis(X)
            dis_loss_real = criterion(output_real, labels_real)

            z_fake = torch.randn(batch_size, 100, 1, 1).cuda()
            X_fake = gen(z_fake)
            output_fake = dis(X_fake)
            dis_loss_fake = criterion(output_fake, labels_fake)

            dis_loss = dis_loss_real + dis_loss_fake

            dis_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            # Generator
            z_fake = torch.randn(batch_size, 100, 1, 1).cuda()
            X_fake = gen(z_fake)
            output_fake = dis(X_fake)
            gen_loss = criterion(output_fake, labels_real)

            dis_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            if idx % 100 == 0:
                print('[{}|{}]: dis loss {} - gen loss {}'.format(
                    epoch, idx, dis_loss.item(), gen_loss.item()))

        gen.eval()
        X_fake = gen(fixed_z)
        gen.train()

        images = X_fake.squeeze().cpu().data.numpy()
        epoch_images = []
        for image_index in range(25):
            image = images[image_index]
            image = image - np.min(image)
            image = image / np.max(image)
            image = (image * 255).astype(np.uint8)
            epoch_images.append(image)

        plt.figure()
        for _idx, each_image in enumerate(epoch_images):
            plt.subplot(5, 5, _idx + 1)
            plt.imshow(each_image, cmap='gray')

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
