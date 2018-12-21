from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dawnet.models.perceive import BaseModel
from dawnet.models.convs import ResidualBottleneckPreactUnit


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


class WideResNet(BaseModel):
    """Wide ResNet implementation using dawnet. The architecture is
    documented here:
        https://arxiv.org/abs/1605.07146
        https://github.com/szagoruyko/wide-residual-networks/

    The Github link provides architecture detail about WideResNet. Basically:
        - depth of the networks should be 6n + 4
        - the number of channels in each convolution is multiple of either
        16, 32 or 64
        - each block uses preact architecture:
                bn -> relu -> conv2d -> bn -> relu -> conv2d -> + id
        - blocks are then grouped:
            - conv0:
            - group0: in 16 channels, out 16x channels, n consec blocks
            - group1: in 16x channels, out 32x channels, n consec blocks
            - group2: in 32x channels, out 64x channels, n consec blocks
            - bn -> relu -> global_avg -> linear -> softmax
    """

    def __init__(self, depth=4, width=1, name=None):
        """Initialize the model"""
        super(WideResNet, self).__init__(name=name)
        self.convs = nn.Sequential()
        self.convs.add_module(
            'conv0',
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1,
                padding=1, bias=False))
        self.convs.add_module(
            'group0',
            self.construct_residual_block(
                in_channels=16, out_channels=16*width, stride=1,
                n_units=depth, unit_type=ResidualBottleneckPreactUnit))
        self.convs.add_module(
            'group1',
            self.construct_residual_block(
                in_channels=16*width, out_channels=32*width, stride=2,
                n_units=depth, unit_type=ResidualBottleneckPreactUnit))
        self.convs.add_module(
            'group2',
            self.construct_residual_block(
                in_channels=32*width, out_channels=64*width, stride=2,
                n_units=depth, unit_type=ResidualBottleneckPreactUnit))
        self.convs.add_module('bn', nn.BatchNorm2d(64*width))
        self.convs.add_module('relu', nn.ReLU())
        self.convs.add_module('gap', nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.linear = nn.Linear(in_features=64*width, out_features=10)

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

    def forward(self, input_data):
        hidden = self.convs(input_data)
        hidden = hidden.view(input_data.size(0), -1)
        return self.linear(hidden)

if __name__ == '__main__':
    model = WideResNet(depth=4, width=1)
    if torch.cuda.is_available():
        model = model.cuda()

    model.x_initialize()
    model.x_train()
