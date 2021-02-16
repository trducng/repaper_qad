import logging
from contextlib import suppress

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from timm.utils import NativeScaler

# from model import ViT
from model import VisionTransformer
from ref_model import ViT
from data import imagenet_train_transform, imagenet_eval_transform


TRAIN_BATCH_SIZE = 256
IMAGENET_TRAIN = '/home/john/john/data/imagenet/train'



def main():

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = 'native'

    model = ViT(image_size=256, patch_size=32, num_classes=1000, dim=768, depth=12,
                heads=12, mlp_dim=3072)
    # model = VisionTransformer()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                          lr=0.003, weight_decay=0.0001)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # vanilla dataloader here
    trainset = datasets.ImageFolder(root=IMAGENET_TRAIN, transform=imagenet_train_transform)
    loader_train = DataLoader(
            trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=16)

    # setup loss function
    train_loss_fn = nn.CrossEntropyLoss().cuda()

    for epoch in range(0, 200):

        train_metrics = train_one_epoch(
            epoch, model, loader_train, optimizer, train_loss_fn,
            amp_autocast=amp_autocast, loss_scaler=loss_scaler)


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn,
        amp_autocast=suppress,
        loss_scaler=None):

    model.train()

    for batch_idx, (input, target) in enumerate(loader):
        input, target = input.cuda(), target.cuda()

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer, clip_grad=None, parameters=model.parameters())
        else:
            loss.backward()
            optimizer.step()

        if batch_idx % 50 == 0:
            print(f'Loss - {epoch}: {loss.item()}')

        if batch_idx % 1000 == 0:
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state_dict, f'logs/train_reduce_{epoch}.pth')

if __name__ == '__main__':
    main()
