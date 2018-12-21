import pdb

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

# ResidualBasicUnit: 50000 -> 0.723
# ResidualBottleneckUnit: 50000 -> 0.5912
# ResidualBasicPreactUnit: 50000 -> 0.7093
# ResidualBottleneckPreactUnit: 50000 -> 0.6087


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


class _ResNetBase(BaseModel):
    """Base class to use the same training code

    The training is on CIFAR10, and has:
        - per pixel mean subtracted
        - first layer is 3x3 convolutions
        - next, 6n layers 3x3 convolutions on feature map of size 32, 16, 8
        - the number of filters are 16, 32, 64
        - subsampling in the convolution with stride of 2
        - global average pooling -> 10-way FC -> softmax

    => The total number of layers are 6n + 2, with n is the number of blocks
    The paper mentions these blocks:
        - 20        -> 3
        - 32        -> 5
        - 44        -> 7
        - 56        -> 9
        - 110       -> 18
        - 1202      -> 200
    """
    def __init__(self, name=None):
        super(_ResNetBase, self).__init__(name)
        self.convs = None
        self.gap = None
        self.linear = None

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError('`forward` should be implemented')


class ResNet20(_ResNetBase):

    def __init__(self, name=None):
        """Initialize ResNet34"""

        super(ResNet20, self).__init__(name)

        self.convs = nn.Sequential()
        self.convs.add_module(
            'conv_initial',
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                      stride=2, padding=0, bias=True))
        self.convs.add_module(
            'conv_block_3_1',
            self.construct_residual_block(
                in_channels=16, out_channels=16, stride=1, n_units=3,
                unit_type=ResidualBottleneckPreactUnit))
        self.convs.add_module(
            'conv_block_3_2',
            self.construct_residual_block(
                in_channels=16, out_channels=32, stride=2, n_units=3,
                unit_type=ResidualBottleneckPreactUnit))
        self.convs.add_module(
            'conv_block_3_3',
            self.construct_residual_block(
                in_channels=32, out_channels=64, stride=2, n_units=3,
                unit_type=ResidualBottleneckPreactUnit))

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(in_features=64, out_features=10, bias=True)

    def forward(self, x):
        """Perform the forward pass"""
        hidden = self.convs(x)
        hidden = self.gap(hidden)
        hidden = hidden.view(x.size(0), -1)
        out = self.linear(hidden)

        return out


if __name__ == '__main__':
    model = ResNet20()

    if torch.cuda.is_available():
        model = model.cuda()

    model.x_initialize()
    model.x_train()
