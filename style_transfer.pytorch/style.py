"""Style transfer reimplementation

- Get a pre-trained encoder
- Get a style image
- Get a content image
- Get a gaussian noise image
- Define style loss and content loss
- Optimize the noise image to lower that loss
"""
from collections import namedtuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.pretrained = models.vgg16(pretrained=True).features
        self.pool_indices = [4, 9, 16, 23, 30]
        self.output = namedtuple(
            "Outputs", ["slice{}".format(idx) for idx in range(len(self.pool_indices))]
        )

    def forward(self, input_x):
        outputs = []
        h = input_x
        last_idx = 0
        for pool_idx in self.pool_indices:
            h = self.pretrained[last_idx:pool_idx](h)
            outputs.append(h)
            last_idx = pool_idx

        return self.output(*outputs)


def prepare_image(np_image):
    image = torch.FloatTensor(np_image) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (image - mean) / std


def get_npimage(image):
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    np_image = (image * std + mean) * 255.0
    np_image = np_image.clamp(0.0, 255.0)
    np_image = np_image.squeeze().permute(1, 2, 0).cpu().data.numpy()
    return np_image.astype(np.uint8)



def gram_matrix(features):
    """Construct the gram matrix for a feature map"""
    batch, channels, height, width = features.shape
    features = features.view(batch, channels, height * width)
    inner_prod = torch.bmm(features, features.transpose(1, 2))
    return inner_prod / (channels * height * width)


if __name__ == "__main__":

    content_weight = 1e5
    style_weight = 1e10

    # get encoder
    encoder = VGG16()
    encoder.eval()
    for each_param in encoder.parameters():
        each_param.requires_grad = False

    # get content & style images
    content_image = cv2.cvtColor(cv2.imread('datasets/content.jpg'), cv2.COLOR_BGR2RGB)
    content_image = prepare_image(content_image)
    style_image = cv2.cvtColor(cv2.imread('datasets/style.jpg'), cv2.COLOR_BGR2RGB)
    style_image = prepare_image(style_image)

    # random image to optimize
    image = torch.randn(*content_image.shape)
    criterion = nn.MSELoss()

    # use gpu
    encoder = encoder.cuda()
    content_image, style_image = content_image.cuda(), style_image.cuda()
    image = image.cuda()

    # optimization
    image = image.requires_grad_()
    optimizer = optim.Adam([image], lr=1e-3)

    style_output = encoder(style_image)
    style_grams = [gram_matrix(each) for each in style_output[:-2]]
    content_output = encoder(content_image)

    for idx in range(30001):
        output = encoder(image)

        grams = [gram_matrix(each) for each in output[:-2]]
        style_loss = 0
        for style1, style2 in zip(grams, style_grams):
            style_loss += criterion(style1, style2)

        content_loss = criterion(output[3], content_output[3])
        loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1000 == 0:
            print(idx, loss.item(), style_loss.item(), content_loss.item())
            im = get_npimage(image)
            cv2.imwrite(f'logs/{idx}.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
