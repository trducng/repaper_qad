from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dawnet.models.perceive import BaseModel
from dawnet.models.convs import ResidualNextUnit

# 50,000 => 0.6966

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


class ResNext(BaseModel):
    """ResNext model as described in this paper:
        https://arxiv.org/abs/1611.05431
    """

    def __init__(self, name=None):
        """Initialize the model"""
        super(ResNext, self).__init__(name=name)
        self.convs = nn.Sequential()
        self.convs.add_module(
            'conv0',
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                      stride=2, padding=0, bias=False))
        self.convs.add_module(
            'resnext0',
            self.construct_residual_block(
                in_channels=64, out_channels=256, stride=1, n_units=3,
                unit_type=ResidualNextUnit, cardinality=32))
        self.convs.add_module(
            'resnext1',
            self.construct_residual_block(
                in_channels=256, out_channels=512, stride=2, n_units=3,
                unit_type=ResidualNextUnit, cardinality=32))
        self.convs.add_module(
            'resnext2',
            self.construct_residual_block(
                in_channels=512, out_channels=1024, stride=2, n_units=3,
                unit_type=ResidualNextUnit, cardinality=32))
        
        self.convs.add_module('gap', nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(1024, 10)
    
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
    model = ResNext()
    if torch.cuda.is_available():
        model = model.cuda()
    model.x_initialize()
    model.x_train()
