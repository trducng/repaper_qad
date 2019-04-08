"""Implement toy VAE on MNIST"""
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image


TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])
BATCH_SIZE = 128
Z_DIM = 20


def get_dataset():
    train_data = datasets.MNIST(
        'datasets', train=True, download=True, transform=TRANSFORM)
    test_data = datasets.MNIST(
        'datasets', train=False, download=True, transform=TRANSFORM)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=25)

    return train_loader, test_loader


class VAE(nn.Module):
    """The Variational Auto-Encoder"""

    def __init__(self):
        """Initialize the object"""
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.Softplus(),
            nn.Linear(in_features=512, out_features=512),
            nn.Softplus()
        )

        self.encoder_mean = nn.Linear(in_features=512, out_features=Z_DIM)
        self.encoder_std = nn.Linear(in_features=512, out_features=Z_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=Z_DIM, out_features=512),
            nn.Softplus(),
            nn.Linear(in_features=512, out_features=512),
            nn.Softplus(),
            nn.Linear(in_features=512, out_features=784),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Perform the encoding
        
        # Argumets
            x [2D tensor]: shape [B, 784]
        
        # Returns
            [2D tensor]: shape [B, Z_DIM]
            [2D tensor]: shape [B, Z_DIM]
        """
        hidden = self.encoder(x)
        return self.encoder_mean(hidden), self.encoder_std(hidden)

    def decode(self, z):
        """Perform the decoding

        # Arguments
            z [2D tensor]: shape [B, Z_DIM]
        
        # Returns
            [2D tensor]: shape [B, 784]
        """
        return self.decoder(z)

    def forward(self, x):
        """Perform the forward pass

        # Arguments
            x [2D tensor]: shape [B, 784]
        
        # Returns
            [2D tensor]: the reconstructed input - shape [B, 784]
            [2D tensor]: the mean - shape [B, Z_DIM]
            [2D tensor]: the logar - shape [B, Z_DIM]
        """
        mean, logvar = self.encode(x)

        eps = torch.randn(x.size(0), Z_DIM).cuda()
        z = mean + eps * torch.exp(0.5 * logvar)       # @TODO: try std directly

        return self.decode(z), mean, logvar


def cost_kl(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar))


def main(folder_name):
    """Perform the training"""

    train_loader, test_loader = get_dataset()
    test_data = iter(test_loader).next()[0].cuda()
    save_image(
        test_data, os.path.join(folder_name, 'original.png'),
        nrow=5, scale_each=True)
    test_data = test_data.view(test_data.size(0), -1)
    model = VAE().cuda()

    cost_reconstruct = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    fixed_z = torch.randn(25, Z_DIM).cuda()

    epochs = 20
    for epoch in range(epochs):
        for idx, (x, _) in enumerate(train_loader):
            x = x.view(x.size(0), -1).contiguous().cuda()
            x_reconstructed, mean, logvar = model(x)

            loss_reconstructed = cost_reconstruct(x_reconstructed, x)
            loss_kl = cost_kl(mean, logvar)
            loss = loss_reconstructed + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 200 == 0:
                print('[{}|{}] loss_reconstructed, loss_kl - {} + {} = {}'
                      .format(epoch, idx, loss_reconstructed.item(),
                              loss_kl.item(), loss.item()))

        # track progress with fixed z
        model.eval()
        outputs = model.decode(fixed_z)
        model.train()

        outputs = outputs.view(25, 28, 28).cpu().data.numpy()
        epoch_images = []
        for idx in range(25):
            image = outputs[idx]
            image = image - np.min(image)
            image = image / np.max(image)
            image = (image * 255).astype(np.uint8)
            epoch_images.append(image)

        plt.figure()
        for _idx, each_image in enumerate(epoch_images):
            ax = plt.subplot(5, 5, _idx + 1)
            ax.imshow(each_image, cmap='gray')
            ax.axis('off')

        plt.suptitle('Epoch {}'.format(epoch))
        plt.savefig(os.path.join(folder_name, '{}.png'.format(epoch)))
        plt.close()

        # track progress with test datatset
        model.eval()
        outputs, mean, logvar = model(test_data)
        loss_reconstructed = cost_reconstruct(x_reconstructed, x)
        loss_kl = cost_kl(mean, logvar)
        loss = loss_reconstructed + loss_kl
        model.train()

        outputs = outputs.view(outputs.size(0), 1, 28, 28)
        save_image(
            outputs,
            os.path.join(folder_name, 'reconstruct_{}.png'.format(epoch)),
            nrow=5, scale_each=True)

        print('====> Epoch {}: kl loss {}, reconstruction loss {}, total loss {}'
            .format(epoch, loss_kl.item(), loss_reconstructed.item(), loss.item()))

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
