from pathlib import Path

import cv2
import numpy as np
import torch

from models import UNet
from transformer_net import TransformerNet


MODEL_PATH = '/data2/information-decomposition/john/logs/style/unet_1e5_1e10.pt.0003'
INPUT_IMAGE_PATH = '/data2/information-decomposition/john/datasets/content/sample.jpg'
OUTPUT_IMAGE_PATH = '/home/john/output6.png'
GPU = True


if GPU:
    chpt = torch.load(MODEL_PATH)
else:
    chpt = torch.load(MODEL_PATH, map_location='cpu')

model = UNet()
# model = TransformerNet()
model.load_state_dict(chpt['model'])
model.eval()

image = cv2.cvtColor(cv2.imread(INPUT_IMAGE_PATH), cv2.COLOR_BGR2RGB)
print('Input shape', image.shape)
image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255

if GPU:
    model = model.cuda()
    image = image.cuda()

pred = model(image)
import pdb; pdb.set_trace()
# output_image = pred.cpu().squeeze().permute(1, 2, 0).data.numpy()       # NOTE: pred might not in range 0-1
# output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
# output_image = (output_image * 255).astype(np.uint8)
output_image = (pred.cpu().squeeze().permute(1, 2, 0).data.numpy() * 255).astype(np.uint8)

cv2.imwrite(OUTPUT_IMAGE_PATH, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
