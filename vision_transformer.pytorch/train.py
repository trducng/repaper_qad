import torch
import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from data import imagenet_train_transform, imagenet_eval_transform
from model import VisionTransformer
from ref_model import ViT

TRAIN_BATCH_SIZE = 4

IMAGENET_TRAIN = '/home/john/john/data/imagenet/train'
IMAGENET_VAL = '/home/john/datasets/imagenet/object_localization/val'
IMAGENET_TEST = '/home/john/datasets/imagenet/object_localization/test'


def main():

    trainset = datasets.ImageFolder(root=IMAGENET_TRAIN, transform=imagenet_train_transform)

    model = VisionTransformer()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                          lr=0.003, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=5, eta_min=0.003)

    epoch = 0
    model.train()
    while True:
        trainloader = DataLoader(
            trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=16)
        for idx, (input_, label) in enumerate(trainloader):
            input_, label = input_.cuda(), label.cuda()
            break
        while True:
            output = model(input_)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 50 == 0:
                print(f'Loss - {epoch}: {loss.item()}')

            # if idx % 1000 == 0:
            #     state_dict = {
            #         'model': model.state_dict(),
            #         'optimizer': optimizer.state_dict()
            #     }
            #     torch.save(state_dict, f'logs/test_{epoch}.pth')
        epoch += 1
        # scheduler.step(epoch)


if __name__ == '__main__':
    main()
