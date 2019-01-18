"""Plot the loss landscape of a single image

This script relies on Pytorch model zoo: as a result, we would need to follow
their preprocessing specification for input image depending on the models:
- all: transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
- resnet: 224 x 224

Some other good things to have to understand more about loss landscape:

x Include prediction accompanying with alpha
x Histogram of prediction
x Move using base direction (a single weight only)
- Move using PCA direction (direction of most influence, only work for several models)
x Export the direction
x Comparing loss landscapes using difference images

The CTC loss is nice in that it allows a crude way to map between sequence
of input to sequence of output without having to have the perfect alignment.
"""

import copy
import math
import os
import pdb

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

from utils import IDX_TO_CLASS


os.makedirs('logs', exist_ok=True)
os.makedirs('datasets', exist_ok=True)
os.environ['TORCH_MODEL_ZOO'] = 'logs'

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def normalize_direction(direction, wab):
    """Normalize the direction vector to have the same norm as
    the weight and bias vector"""
    return direction / direction.norm() * wab.norm()
    # return direction.mul_(wab.norm() / direction.norm() + 1e-10)


def get_random_direction(state_dict):
    """Get random direction, sampled from a Gaussian distribution"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if value.dim() <= 1:
            # for empty and scalar value
            norm_dir = value
        else:
            random_dir = torch.randn(value.shape)
            norm_dir = normalize_direction(random_dir, value)
            # print(random_dir.norm(), value.norm(), norm_dir.norm())
        new_state_dict[key] = norm_dir

    return new_state_dict


def get_random_base_direction(state_dict, layer=None, position=None):
    """Get random direction, basically a very sparse vector, positive/negative
    in 1 place and 0 everywhere"""
    new_state_dict = {}
    layer_keys = [key for key, value in state_dict.items()
                  if isinstance(value, torch.Tensor) and value.dim() > 1]

    if layer is None:
        layer = np.random.choice(layer_keys)

    for idx, (name, value) in enumerate(state_dict.items()):
        if isinstance(layer, int) and layer != idx:
            new_state_dict[name] = value
            continue
        elif isinstance(layer, str) and layer != name:
            new_state_dict[name] = value
            continue

        tensor = copy.deepcopy(value)
        if position is None:
            shape = tensor.shape
            position = tuple([np.random.randint(each) for each in shape])

        tensor[position] = tensor.norm()
        new_state_dict[name] = tensor

    return new_state_dict


def get_difference_state(state1, state2):
    """Get state1 - state2"""
    diff_state = {}
    for name, value in state1.items():
        diff_state[name] = value - state2[name]

    return diff_state


def update_state_1d(state_dict, diff_state, alpha):
    """Return state_dict + diff_state * alpha"""
    new_state = {}
    for key, value in state_dict.items():
        new_state[key] = value + diff_state[key] * alpha

    return new_state


def update_state_2d(state_dict, diff1, diff2, alpha1, alpha2):
    """Return state_dict + diff1 * alpha1 + diff2 * alpha2"""
    new_state = {}
    for key, value in state_dict.items():
        new_state[key] = value + diff1[key] * alpha1 + diff2[key] * alpha2

    return new_state


def plot_single_image_1d(
    image_path='images/tiger_incorrect.png', size=224, class_=292, debug=True):
    """Given a trained model and an image, draw the qqloss landscape of the
    model weight along a random 1D weight direction.

    For tiger, the correct class is 292.
    """

    # setup model and losses
    model = models.resnet18(pretrained=True)
    model.eval()
    loss = nn.CrossEntropyLoss()

    # setup the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))

    image_torch = NORMALIZE(torch.FloatTensor(image / 255))
    image_torch = image_torch.transpose(0, 2).transpose(1, 2).unsqueeze(0)

    # set up the linear interpolation
    weight = copy.deepcopy(model.state_dict())  # w/o deepcopy, `weight` will change
    direction = get_random_direction(weight)
    diff = get_difference_state(direction, weight)

    # gather the losses
    alpha = list((np.arange(100)-50) / 100)
    # alpha = list((np.arange(10)-5) / 10)
    losses = []
    debugs = []
    for each_alpha in tqdm(alpha):
        # each_alpha = 0.5
        inter_weight = update_state_1d(weight, diff, each_alpha)
        model.load_state_dict(inter_weight)
        model.eval()
        with torch.no_grad():
            # 292
            pred = model(image_torch).detach()
            l = loss(pred, torch.tensor([class_], dtype=torch.long)).detach().item()
            if debug:
                pred = IDX_TO_CLASS[np.argmax(pred.data.numpy())]
                print_debug = '{}, {}, {}'.format(each_alpha, l, pred)
                tqdm.write(print_debug)
                debugs.append(print_debug)
        losses.append(l)


    fig = plt.figure()
    plt.plot(alpha, losses)
    fig.canvas.draw()

    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    cv2.imwrite(os.path.join('logs', '1d_{}_'.format(class_) + os.path.basename(image_path)),
                cv2.cvtColor(plot, cv2.COLOR_RGB2BGR))
    fig.show()

    if debug:
        with open(os.path.join('logs', '1d_{}_.txt'.format(class_)), 'w') as f_out:
            f_out.write('\n'.join(debugs))


def plot_single_image_2d(
    image_path='images/tiger_incorrect.png', class_=292, size=224, debug=True,
    save_diff=True):
    """Given a trained model and an image, draw the loss landscape of the
    model weight along a random 2D weight direction"""

    # setup model and losses
    model = models.resnet18(pretrained=True)
    model.eval()
    loss = nn.CrossEntropyLoss()

    # setup the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))

    image_torch = NORMALIZE(torch.FloatTensor(image / 255))
    image_torch = image_torch.transpose(0, 2).transpose(1, 2).unsqueeze(0)

    # set up the linear interpolation
    weight = copy.deepcopy(model.state_dict())  # w/o deepcopy, `weight` will change
    direction1, direction2 = get_random_direction(weight), get_random_direction(weight)
    diff1 = get_difference_state(direction1, weight)
    diff2 = get_difference_state(direction2, weight)

    with save_diff:
        obj = {
            'diff1': diff1,
            'diff2': diff2
        }
        torch.save(obj, os.path.join('logs/2d.john'))

    # gather the losses
    alpha1 = list((np.arange(11)-5) / 10)
    alpha2 = list((np.arange(11)-5) / 10)
    # alpha1 = list((np.arange(110)-50) / 100)
    # alpha2 = list((np.arange(110)-50) / 100)
    X, Y = np.meshgrid(alpha1, alpha2)
    losses = []
    _X, _Y = X.ravel(), Y.ravel()
    debugs = []

    for each_alpha1, each_alpha2 in tqdm(list(zip(_X, _Y))):
        inter_weight = update_state_2d(weight, diff1, diff2, each_alpha1, each_alpha2)
        model.load_state_dict(inter_weight)
        model.eval()
        with torch.no_grad():
            pred = model(image_torch).detach()
            l = loss(pred, torch.tensor([class_], dtype=torch.long)).detach().item()
            l = math.log(l, 10)
            if debug:
                pred = np.argmax(pred.data.numpy())
                pred_string = IDX_TO_CLASS[pred]
                print_debug = '{}, {}, {}, {}, {}'.format(each_alpha1, each_alpha2, l, pred_string, pred)
                tqdm.write(print_debug)
                debugs.append(print_debug)
        losses.append(l)

    losses = np.array(losses).reshape(X.shape)
    fig = plt.figure()
    plt.contourf(X, Y, losses, cmap='RdGy')
    plt.colorbar()
    fig.canvas.draw()

    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    cv2.imwrite(os.path.join('logs', '2d_{}_'.format(class_) + os.path.basename(image_path)),
                cv2.cvtColor(plot, cv2.COLOR_RGB2BGR))
    fig.show()

    if debug:
        with open(os.path.join('logs', '2d_{}.txt'.format(class_)), 'w') as f_out:
            f_out.write('\n'.join(debugs))
        with open('tmp.txt', 'w') as f_out:
            _tmp = [each.split(',')[-1].strip() for each in debugs]
            f_out.write('\n'.join(_tmp))


def plot_interpolate_image(
        image1_path='images/tiger_correct.png',
        image2_path='images/tiger_incorrect.png', size=224, debug=True):
    """This is very the same as the above, but interpolate between 2 image
    points rather than 2 weight points"""

    # setup model and loss
    model = models.resnet18(pretrained=True)
    model.eval()
    loss = nn.CrossEntropyLoss()

    # get image1
    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = cv2.resize(image1, (size, size))

    image1_torch = NORMALIZE(torch.FloatTensor(image1 / 255))
    image1_torch = image1_torch.transpose(0, 2).transpose(1, 2).unsqueeze(0)

    # get image2
    image2 = cv2.imread(image2_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image2, (size, size))

    image2_torch = NORMALIZE(torch.FloatTensor(image2 / 255))
    image2_torch = image2_torch.transpose(0, 2).transpose(1, 2).unsqueeze(0)

    # get the difference
    image_diff_torch = image2_torch - image1_torch

    losses = []
    alpha = list(np.linspace(-1.5, 1.5, 40))
    # alpha = [-1, -0.5, 0, 0.5, 1]
    for each_alpha in tqdm(alpha):
        image_inter = image1_torch + each_alpha * image_diff_torch
        with torch.no_grad():
            pred = model(image_inter).detach()
            l = loss(pred, torch.tensor([292], dtype=torch.long)).detach().item()
            if debug:
                pred = IDX_TO_CLASS[np.argmax(pred.data.numpy())]
                tqdm.write('{}, {}, {}'.format(each_alpha, l, pred))
        losses.append(l)

    fig = plt.figure()
    plt.plot(alpha, losses)
    fig.canvas.draw()

    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    cv2.imwrite(
        os.path.join('logs', 'inter.png'),
        cv2.cvtColor(plot, cv2.COLOR_RGB2BGR))


def plot_test(
    image_path='images/tiger_incorrect.png', size=224, class_=292, debug=True):
    """Given a trained model and an image, draw the qqloss landscape of the
    model weight along a random 1D weight direction.

    For tiger, the correct class is 292.
    """

    # setup model and losses
    model = models.resnet18(pretrained=True)
    model.eval()
    loss = nn.CrossEntropyLoss()

    # setup the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))

    image_torch = NORMALIZE(torch.FloatTensor(image / 255))
    image_torch = image_torch.transpose(0, 2).transpose(1, 2).unsqueeze(0)

    weight = copy.deepcopy(model.state_dict())  # w/o deepcopy, `weight` will change
    predictions = []

    for _ in range(5):
        # set up the linear interpolation
        direction = get_random_direction(weight)
        diff = get_difference_state(direction, weight)

        # gather the losses
        alpha = list((np.arange(100)-50) / 100)
        # alpha = list((np.arange(10)-5) / 10)
        losses = []
        debugs = []
        preds = []
        for each_alpha in tqdm(alpha):
            # each_alpha = 0.5
            inter_weight = update_state_1d(weight, diff, each_alpha)
            model.load_state_dict(inter_weight)
            model.eval()
            with torch.no_grad():
                # 292
                pred = model(image_torch).detach()
                l = loss(pred, torch.tensor([class_], dtype=torch.long)).detach().item()
                if debug:
                    pred = IDX_TO_CLASS[np.argmax(pred.data.numpy())]
                    preds.append(pred)
                    print_debug = '{}, {}, {}'.format(each_alpha, l, pred)
                    tqdm.write(print_debug)
                    debugs.append(print_debug)
            losses.append(l)
        predictions.append(preds)

    for idx, p in enumerate(predictions):
        with open('tmp_{}.txt'.format(idx), 'w') as f_out:
            f_out.write('\n'.join(p))


if __name__ == '__main__':
    plot_test(debug=True)
    # plot_interpolate_image()