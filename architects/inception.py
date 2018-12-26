from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dawnet.models.perceive import BaseModel

# 50,000 -> 0.8081


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


class BaseConvolution(nn.Module):
    """Basic convolution structure"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        """Initialize the basic convolution layer coupled with batchnorm"""
        super(BaseConvolution, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input_x):
        """Perform the forward pass"""
        hidden = self.conv(input_x)
        hidden = self.bn(hidden)
        output = F.relu(hidden, inplace=True)

        return output


class InceptionA(nn.Module):
    """InceptionA implementation
        https://arxiv.org/pdf/1602.07261.pdf
    """
    def __init__(self, in_channels, out_channels):
        """Initialize the inception module version A"""
        super(InceptionA, self).__init__()

        # get the number of output channels
        out = out_channels // 4
        out_pool = out
        if out_channels % 4 != 0:
            out_pool = out_channels - 3 * out
        out_inter = out // 2

        # branch 1
        self.b_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BaseConvolution(
                in_channels=in_channels, out_channels=out_pool, kernel_size=1))

        # branch 2
        self.b_2 = BaseConvolution(
            in_channels=in_channels, out_channels=out, kernel_size=1)

        # branch 3
        self.b_3 = nn.Sequential(
            BaseConvolution(
                in_channels=in_channels, out_channels=out_inter,
                kernel_size=1),
            BaseConvolution(
                in_channels=out_inter, out_channels=out, kernel_size=3,
                padding=1))

        # branch 4
        self.b_4 = nn.Sequential(
            BaseConvolution(
                in_channels=in_channels, out_channels=out_inter,
                kernel_size=1),
            BaseConvolution(
                in_channels=out_inter, out_channels=out, kernel_size=3,
                padding=1),
            BaseConvolution(
                in_channels=out, out_channels=out, kernel_size=3,
                padding=1))

    def forward(self, input_x):
        """Perform the forward pass"""
        b_1 = self.b_1(input_x)
        b_2 = self.b_2(input_x)
        b_3 = self.b_3(input_x)
        b_4 = self.b_4(input_x)
        return torch.cat([b_1, b_2, b_3, b_4], dim=1)


class InceptionB(nn.Module):
    """InceptionB-v4 implementation"""

    def __init__(self, in_channels, out_channels):
        """Initialize the object"""
        super(InceptionB, self).__init__()

        # get the number of output channels
        out = out_channels // 4
        out_pool = out
        if out_channels % 4 != 0:
            out_pool = out_channels - 3 * out
        out_inter = out // 2

        # branch 1
        self.b_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BaseConvolution(
                in_channels=in_channels, out_channels=out_pool, kernel_size=1)
        )

        # branch 2
        self.b_2 = BaseConvolution(
            in_channels=in_channels, out_channels=out, kernel_size=1)

        # branch 3
        self.b_3 = nn.Sequential(
            BaseConvolution(
                in_channels=in_channels, out_channels=out_inter,
                kernel_size=1),
            BaseConvolution(
                in_channels=out_inter, out_channels=out_inter,
                kernel_size=(1, 7), padding=(0, 3)),
            BaseConvolution(
                in_channels=out_inter, out_channels=out, kernel_size=(7, 1),
                padding=(3, 0)))

        # branch 4
        self.b_4 = nn.Sequential(
            BaseConvolution(
                in_channels=in_channels, out_channels=out_inter,
                kernel_size=1),
            BaseConvolution(
                in_channels=out_inter, out_channels=out_inter,
                kernel_size=(1, 7), padding=(0, 3)),
            BaseConvolution(
                in_channels=out_inter, out_channels=out_inter,
                kernel_size=(7, 1), padding=(3, 0)),
            BaseConvolution(
                in_channels=out_inter, out_channels=out_inter,
                kernel_size=(1, 7), padding=(0, 3)),
            BaseConvolution(
                in_channels=out_inter, out_channels=out, kernel_size=(7, 1),
                padding=(3, 0)))

    def forward(self, input_x):
        """Perform the forward pass"""
        b_1 = self.b_1(input_x)
        b_2 = self.b_2(input_x)
        b_3 = self.b_3(input_x)
        b_4 = self.b_4(input_x)
        return torch.cat([b_1, b_2, b_3, b_4], dim=1)


class InceptionC(nn.Module):
    """Implement the InceptionC-v4"""

    def __init__(self, in_channels, out_channels):
        """Initialize the object"""
        super(InceptionC, self).__init__()

        # get the number of output channels
        out = out_channels // 6
        out_pool = out
        if out_channels % 6 != 0:
            out_pool = out_channels - 5 * out
        out_inter = out // 2

        # branch 1
        self.b_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BaseConvolution(
                in_channels=in_channels, out_channels=out_pool, kernel_size=1))

        # branch 2
        self.b_2 = BaseConvolution(
            in_channels=in_channels, out_channels=out, kernel_size=1)

        # branch 3
        self.b3_conv11 = BaseConvolution(
            in_channels=in_channels, out_channels=out_inter, kernel_size=1)
        self.b3_conv13 = BaseConvolution(
            in_channels=out_inter, out_channels=out, kernel_size=(1, 3),
            padding=(0, 1))
        self.b3_conv31 = BaseConvolution(
            in_channels=out_inter, out_channels=out, kernel_size=(3, 1),
            padding=(1, 0))

        # branch 4
        self.b4_conv11 = BaseConvolution(
            in_channels=in_channels, out_channels=out_inter, kernel_size=1)
        self.b4_conv13_1 = BaseConvolution(
            in_channels=out_inter, out_channels=out_inter, kernel_size=(1, 3),
            padding=(0, 1))
        self.b4_conv31_2 = BaseConvolution(
            in_channels=out_inter, out_channels=out_inter, kernel_size=(3, 1),
            padding=(1, 0))
        self.b4_conv31_3 = BaseConvolution(
            in_channels=out_inter, out_channels=out, kernel_size=(3, 1),
            padding=(1, 0))
        self.b4_conv13_3 = BaseConvolution(
            in_channels=out_inter, out_channels=out, kernel_size=(1, 3),
            padding=(0, 1))

    def forward(self, input_x):
        """Perform the forward pass"""
        b_1 = self.b_1(input_x)
        b_2 = self.b_2(input_x)

        b_3 = self.b3_conv11(input_x)
        b_3_1 = self.b3_conv13(b_3)
        b_3_2 = self.b3_conv31(b_3)

        b_4 = self.b4_conv11(input_x)
        b_4 = self.b4_conv13_1(b_4)
        b_4 = self.b4_conv31_2(b_4)
        b_4_1 = self.b4_conv31_3(b_4)
        b_4_2 = self.b4_conv13_3(b_4)

        return torch.cat([b_1, b_2, b_3_1, b_3_2, b_4_1, b_4_2], dim=1)


class InceptionReductionA(nn.Module):
    """Implement the ReductionA"""
    def __init__(self, in_channels, out_channels):
        """Initialize the object"""
        super(InceptionReductionA, self).__init__()

        # get the number of output channels
        out = (out_channels - in_channels) // 2
        out_v = out
        if (out_channels - in_channels) % 2 != 0:
            out_v = out_channels - in_channels - out
        out_inter = out // 2

        # branch 1
        self.b_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # branch 2
        self.b_2 = BaseConvolution(
            in_channels=in_channels, out_channels=out_v, kernel_size=3,
            stride=2, padding=1)

        # branch 3
        self.b_3 = nn.Sequential(
            BaseConvolution(in_channels=in_channels, out_channels=out_inter,
                            kernel_size=1),
            BaseConvolution(in_channels=out_inter, out_channels=out_inter,
                            kernel_size=3, padding=1),
            BaseConvolution(in_channels=out_inter, out_channels=out,
                            kernel_size=3, stride=2, padding=1))

    def forward(self, input_x):
        """Perform the forward pass"""
        b_1 = self.b_1(input_x)
        b_2 = self.b_2(input_x)
        b_3 = self.b_3(input_x)
        return torch.cat([b_1, b_2, b_3], dim=1)


class InceptionReductionB(nn.Module):
    """Implement ReductionB"""
    def __init__(self, in_channels, out_channels):
        super(InceptionReductionB, self).__init__()

        # get the number of output channels
        out = (out_channels - in_channels) // 2
        out_v = out
        if (out_channels - in_channels) % 2 != 0:
            out_v = out_channels - in_channels - out
        out_inter = out // 2

        # branch 1
        self.b_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # branch 2
        self.b_2 = nn.Sequential(
            BaseConvolution(in_channels=in_channels, out_channels=out_inter,
                            kernel_size=1),
            BaseConvolution(in_channels=out_inter, out_channels=out_v,
                            kernel_size=3, stride=2, padding=1))

        # branch 3
        self.b_3 = nn.Sequential(
            BaseConvolution(in_channels=in_channels, out_channels=out_v,
                            kernel_size=1),
            BaseConvolution(in_channels=out_v, out_channels=out_v,
                            kernel_size=(1, 7), padding=(0, 3)),
            BaseConvolution(in_channels=out_v, out_channels=out_v,
                            kernel_size=(7, 1), padding=(3, 0)),
            BaseConvolution(in_channels=out_v, out_channels=out, kernel_size=3,
                            stride=2, padding=1))

    def forward(self, input_x):
        """Implement the forward pass"""
        b_1 = self.b_1(input_x)
        b_2 = self.b_2(input_x)
        b_3 = self.b_3(input_x)
        return torch.cat([b_1, b_2, b_3], dim=1)


class InceptionV4(BaseModel):
    """Implementation of Inception-v4 on the CIFAR10 dataset. The detail of
    Inceptionv4 can be referenced here:
        https://arxiv.org/abs/1602.07261
        The paper implement inception for ImageNet, since we work on CIFAR,
        we will use a lighter version of it.
    """

    def __init__(self, name=None):
        """Initialize Inception-v3"""
        super(InceptionV4, self).__init__(name=name)

        self.convs = nn.Sequential(
            BaseConvolution(in_channels=3, out_channels=64, kernel_size=3,
                            padding=1),
            InceptionA(in_channels=64, out_channels=96),
            InceptionReductionA(in_channels=96, out_channels=128),
            InceptionB(in_channels=128, out_channels=256),
            InceptionReductionB(in_channels=256, out_channels=384),
            InceptionC(in_channels=384, out_channels=512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.linear = nn.Linear(in_features=512, out_features=10)

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
    model = InceptionV4()
    if torch.cuda.is_available():
        model = model.cuda()
    model.x_initialize()
    model.x_train()
