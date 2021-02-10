import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from data import imagenet_train_transform, imagenet_eval_transform
from model import ViT

TRAIN_BATCH_SIZE = 4

IMAGENET_TRAIN = '/home/john/datasets/imagenet/object_localization/train'
IMAGENET_VAL = '/home/john/datasets/imagenet/object_localization/val'
IMAGENET_TEST = '/home/john/datasets/imagenet/object_localization/test'


def main():

    trainset = datasets.ImageFolder(root=IMAGENET_TRAIN, transform=imagenet_train_transform)
    trainloader = DataLoader(
        trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)

    model = ViT(image_size=256, patch_size=32, num_classes=1000, dim=768, depth=12,
                heads=12, mlp_dim=3072)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    model.train()
    for input_, label in trainloader:
        input_, label = input_.cuda(), label.cuda()
        output = model(input_)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}')


if __name__ == '__main__':
    main()
