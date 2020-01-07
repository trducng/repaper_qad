# Simple training and running
import time
from pathlib import Path

import cv2
import fire
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


def train(style_image_path, image_folder, output_folder, filename, n_epochs, gpu):
    image_manipulation = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )

    dataset = datasets.ImageFolder(image_folder, image_manipulation)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, drop_last=True
    )

    model_loss = VGGFeatures(pretrained=True, requires_grad=False)
    # transform_network = UNet(pretrained=False)
    transform_network = TransformerNet()
    if gpu:
        model_loss = model_loss.cuda()
        transform_network = transform_network.cuda()

    style_criterion = nn.MSELoss()
    content_criterion = nn.MSELoss()

    optimizer = optim.Adam(transform_network.parameters(), 1e-3)

    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    ys = cv2.cvtColor(cv2.imread(style_image_path), cv2.COLOR_BGR2RGB)
    ys = torch.FloatTensor(ys).permute(2, 0, 1).unsqueeze(0)
    ys = torch.cat([ys] * 2, dim=0)
    if gpu:
        ys = ys.cuda()
        mean = mean.cuda()
        std = std.cuda()

    pred_ys = normalize_imagenet_image(ys, mean, std)
    pred_ys = model_loss(pred_ys)
    gram_ys = [gram_matrix(each) for each in pred_ys]

    epoch = 0
    while True:
        if epoch >= n_epochs:
            break

        total_loss = 0.0
        current_time = int(time.time())
        for idx, (x, _) in enumerate(dataloader):
            if gpu:
                x = x.cuda()
            yhat = transform_network(x)
            yhat_transform = normalize_imagenet_image(yhat, mean, std)
            x_transform = normalize_imagenet_image(x, mean, std)

            pred_yhat = model_loss(yhat_transform)
            pred_x = model_loss(x_transform)
            content_loss = content_criterion(pred_yhat.relu3, pred_x.relu3)

            gram_yhat = [gram_matrix(each) for each in pred_yhat]
            style_loss = 0
            for each_yhat, each_ys in zip(gram_yhat, gram_ys):
                style_loss += style_criterion(each_yhat, each_ys)

            loss = content_loss + style_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if idx % 1000 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": transform_network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    Path(output_folder, "{}.{:04d}".format(filename, epoch)),
                )
                print(
                    f"[{idx}|{len(dataset)}] Loss {total_loss / (idx + 1)} "
                    f"Take {int(time.time()) - current_time} seconds"
                )
                current_time = int(time.time())

        torch.save(
            {
                "epoch": epoch,
                "model": transform_network.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            Path(output_folder, "{}.{:04d}".format(filename, epoch)),
        )
        epoch += 1


class CLI:
    def train(
        self,
        style_image_path,
        image_folder="datasets/coco",
        output_folder="logs",
        filename="model.pt",
        n_epochs=5,
        gpu=True,
    ):
        train(style_image_path, image_folder, output_folder, filename, n_epochs, gpu)


if __name__ == "__main__":
    fire.Fire(CLI)
