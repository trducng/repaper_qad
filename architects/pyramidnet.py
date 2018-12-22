import math

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dawnet.models.perceive import BaseModel
from dawnet.models.convs import (
    ResidualBasicUnit, ResidualBottleneckUnit,
    ResidualBasicPreactUnit, ResidualBottleneckPreactUnit)


# ResidualBottleneckPreactUnit, additive: 50,000 => 0.6366
# ResidualBottleneckPreactUnit, multiplicative: 50,000 => 
# ResidualBasicPreactUnit, additive: 50,000 =>
# ResidualBottleneckPreactUnit, additive, zero-pad: 50,000 =>
# ResidualBottleneckPreactUnit, additive, zero-pad, neoblock: 50,000 =>
# ResidualBasicPreactUnit, multiplicative, zero-pad, neoblock: 50,000 =>  

TRANSFORM_TRAIN = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

TRANSFORM_TEST = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class PyramidNet(BaseModel):
    """Implement the PyramidNet architecture as documented here:
        https://arxiv.org/abs/1610.02915
        https://github.com/jhkim89/PyramidNet

    In PyramidNet, the number of channels gradually increases, instead of
    maintaining the dimension then sudden jumps. The authors propose to
    increase the number of feature maps under this scheme:
        D_k =   16                              for k = 1
            =   math.floor(D_k-1 + alpha / N)   for 2 <= k <= N + 1
        in which N denotes the total number of residual units (e.g,
            `ResidualBasicUnit`, `ResidualBottleneckUnit`...)

    The above scheme is called additive scheme, another scheme is
    multiplicative scheme:
        D_k =   16                                  for k = 1
            =   math.floor(D_k-1 * alpha**(1/N))    for 2 <= k <= N + 1

    # Arguments
        alpha [float]: the widening factor
        N [int]: the number of residual blocks inside the model
        mul [bool]: whether to use multiplicative or additive scheme
    """

    def __init__(self, alpha, depth, multiplicative, bottleneck, name=None):
        """Initialize the network"""
        super(PyramidNet, self).__init__(name=name)
        self.convs = nn.Sequential()
        self.convs.add_module(
            'conv0',
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1,
                      padding=1, bias=False))

        if bottleneck:
            unit_type = ResidualBottleneckPreactUnit
            n_units = math.floor((depth - 2) / 9)
        else:
            unit_type = ResidualBasicPreactUnit
            n_units = math.floor((depth - 2) / 6)

        # add the first block
        block0, channels = self.construct_pyramid_block(
            in_channels=16, alpha=alpha, depth=depth, stride=1,
            n_units=n_units, unit_type=unit_type,
            multiplicative=multiplicative, zero_pad=True,
            skip_first_relu=True, last_bn=True)
        self.convs.add_module('pyramidblock0', block0)

        # add the second block
        block1, channels = self.construct_pyramid_block(
            in_channels=channels, alpha=alpha, depth=depth, stride=2,
            n_units=n_units, unit_type=unit_type,
            multiplicative=multiplicative, zero_pad=True,
            skip_first_relu=True, last_bn=True)
        self.convs.add_module('pyramidblock1', block1)

        # add the third block
        block2, channels = self.construct_pyramid_block(
            in_channels=channels, alpha=alpha, depth=depth, stride=2,
            n_units=10, unit_type=ResidualBottleneckPreactUnit,
            multiplicative=multiplicative, zero_pad=True,
            skip_first_relu=True, last_bn=True)
        self.convs.add_module('pyramidblock2', block2)

        self.convs.add_module('bn', nn.BatchNorm2d(channels))
        self.convs.add_module('relu', nn.ReLU(inplace=True))
        self.convs.add_module('gap', nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(in_features=channels, out_features=10)

    def forward(self, input_x):
        """Perform the forward pass"""
        hidden = self.convs(input_x)
        hidden = hidden.view(input_x.size(0), -1)
        output = self.linear(hidden)

        return output

    def x_initialize(self):
        """Set up the optimizer"""
        self.iteration = 0
        self.running_loss = 0

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        self.train_data = torchvision.datasets.CIFAR10(
            'datasets', train=True, download=True, transform=TRANSFORM_TRAIN)
        self.test_data = torchvision.datasets.CIFAR10(
            'datasets', train=False, download=True, transform=TRANSFORM_TEST)
        self.train_loader = iter(torch.utils.data.DataLoader(
            self.train_data, batch_size=8))
        self.test_loader = iter(torch.utils.data.DataLoader(
            self.test_data, batch_size=16))

    def x_learn(self):
        self.iteration += 1

        try:
            x, y = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(
                torch.utils.data.DataLoader(self.train_data, batch_size=4))
            x, y = next(self.train_loader)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        preds = self(x)
        cost = self.criterion(preds, y)

        self.optimizer.zero_grad()

        cost.backward()
        self.optimizer.step()

        if self.iteration % 500 == 0:
            print('Iteration {}: {}'.format(self.iteration, self.running_loss/500))
            self.running_loss = cost.item()
            self.x_validate()
            self.x_save('logs')
        else:
            self.running_loss += cost.item()

    def x_train(self):
        print('Begin training')
        while True:
            self.x_learn()

    def x_infer(self, image):
        if isinstance(image, str):
            image = Image.open(image)

        image = TRANSFORM_TEST(image)
        image = image.unsqueeze(0)

        if torch.cuda.is_available():
            image = image.cuda()

        return self(image).item()

    def x_test(self):
        self.test_loader = iter(torch.utils.data.DataLoader(
            self.test_data, batch_size=16))
        correct = 0
        total = 0
        for X, y in self.test_loader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            pred = self(X)
            _, pred = torch.max(pred, 1)
            c = torch.sum((pred == y).squeeze()).item()
            correct += c
            total += X.size(0)

        print('Accuracy: {}'.format(correct / total))
        return correct / total

    def x_validate(self):
        self.test_loader = iter(torch.utils.data.DataLoader(
            self.test_data, batch_size=16))
        correct = 0
        total = 0
        for X, y in self.test_loader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            pred = self(X)
            _, pred = torch.max(pred, 1)
            c = torch.sum((pred == y).squeeze()).item()
            correct += c
            total += X.size(0)

        print('Accuracy: {}'.format(correct / total))
        return correct / total

    def get_progress(self):
        return {}

    def get_save_state(self):
        return {}


if __name__ == '__main__':
    model = PyramidNet(alpha=40, depth=92, multiplicative=False,
                       bottleneck=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.x_initialize()
    model.x_train()
