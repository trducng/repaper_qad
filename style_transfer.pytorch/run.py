# Simple training and running
import time
from pathlib import Path

import cv2
import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import VGGFeatures, UNet
from transformer_net import TransformerNet


def gram_matrix(features):
    """Construct the gram matrix for a feature map"""
    batch, channels, height, width = features.shape
    features = features.view(batch, channels, height * width)
    inner_prod = torch.bmm(features, features.transpose(1, 2))
    return inner_prod / (channels * height * width)


def normalize_imagenet_image(image, mean, std):
    """Normalize image

    # Arguments
        image [4D tensor]: image of shape B, C, height, width

    # Returns
        [4D tensor]: image of shape B, C, height, width
    """
    return (image - mean) / std


def train(model_type, style_image_path, image_folder, output_folder, filename, n_epochs, gpu, seed, load_path):
    """Perform the training phase"""
    ckpt = None
    if isinstance(load_path, str):
        ckpt = torch.load(load_path)
        print(f'Load from {load_path}')

    if ckpt:
        seed = ckpt.get('seed', int(np.random.random() * 1e9))
    elif not isinstance(seed, int):
        seed = int(np.random.random() * 1e9)

    print(f'Using seed {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)

    image_manipulation = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )

    dataset = datasets.ImageFolder(image_folder, image_manipulation)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, drop_last=True
    )

    model_loss = VGGFeatures(pretrained=True, requires_grad=False)
    if model_type == 'unet':
        transform_network = UNet(pretrained=False)
    elif model_type == 'transformer':
        transform_network = TransformerNet()
    if ckpt:
        transform_network.load_state_dict(ckpt['model'])

    if gpu:
        model_loss = model_loss.cuda()
        transform_network = transform_network.cuda()

    style_criterion = nn.MSELoss()
    content_criterion = nn.MSELoss()

    optimizer = optim.Adam(transform_network.parameters(), 1e-3)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    ys = cv2.cvtColor(cv2.imread(style_image_path), cv2.COLOR_BGR2RGB)
    ys = torch.FloatTensor(ys).permute(2, 0, 1).unsqueeze(0) / 255      # NOTE: add / 255
    ys = torch.cat([ys] * 2, dim=0)
    if gpu:
        ys = ys.cuda()
        mean = mean.cuda()
        std = std.cuda()

    pred_ys = normalize_imagenet_image(ys, mean, std)
    pred_ys = model_loss(pred_ys)
    gram_ys = [gram_matrix(each) for each in pred_ys]

    epoch = 0
    if ckpt:
        epoch = ckpt['epoch']
    while True:
        if epoch >= n_epochs:
            break

        total_loss = 0.0
        total_content_loss = 0.0
        total_style_loss = 0.0
        current_time = int(time.time())
        for idx, (x, _) in enumerate(dataloader):
            if gpu:
                x = x.cuda()
            yhat = transform_network(x)
            if model_type == 'transformer':
                # yhat = (yhat - yhat.min()) / (yhat.max() - yhat.min())      # normalize to range 0 - 1
                yhat = torch.sigmoid(yhat)
            yhat_transform = normalize_imagenet_image(yhat, mean, std)
            x_transform = normalize_imagenet_image(x, mean, std)

            pred_yhat = model_loss(yhat_transform)
            pred_x = model_loss(x_transform)
            content_loss = content_criterion(pred_yhat.relu3, pred_x.relu3)

            gram_yhat = [gram_matrix(each) for each in pred_yhat]
            style_loss = 0
            for each_yhat, each_ys in zip(gram_yhat, gram_ys):
                style_loss += style_criterion(each_yhat, each_ys)

            loss = 1e5 * content_loss + 1e10 * style_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_content_loss += content_loss.item()
            total_style_loss += style_loss.item()
            if idx % 1000 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": transform_network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "seed": seed,
                    },
                    Path(output_folder, "{}.{:04d}".format(filename, epoch)),
                )
                print(
                    f"[{idx}|{len(dataset)/2}] Loss {total_loss / (idx + 1)} "
                    f"(style {total_style_loss / (idx + 1)} "
                    f"content {total_content_loss / (idx + 1)}) "
                    f"Take {int(time.time()) - current_time} seconds"
                )
                current_time = int(time.time())

        torch.save(
            {
                "epoch": epoch,
                "model": transform_network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "seed": seed,
            },
            Path(output_folder, "{}.{:04d}".format(filename, epoch)),
        )
        epoch += 1


class CLI:
    def train(
        self,
        model_type,
        style_image_path,
        image_folder="datasets/coco",
        output_folder="logs",
        filename="model.pt",
        n_epochs=5,
        gpu=True,
        seed=None,
        load_path=None,
    ):
        train(model_type, style_image_path, image_folder, output_folder, filename,
              n_epochs, gpu, seed, load_path)


if __name__ == "__main__":
    fire.Fire(CLI)
