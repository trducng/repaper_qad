# Training model with subspace (using dataset provided by Uber)
# @author: _john
# @TODO: what happens if we shuffle with some fixed rules for all images?
# =============================================================================
import argparse
import pdb

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

from models import FullyConnectedSubspace, LeNetSubspace


def train_fc(args):
    # data loading
    train_h5 = h5py.File(args.train_data, 'r')
    train_x = np.array(train_h5['images'])
    train_y = np.array(train_h5['labels'])

    val_h5 = h5py.File(args.val_data, 'r')
    val_x = np.array(val_h5['images'])
    val_y = np.array(val_h5['labels'])

    # mild data processing
    train_x = train_x.squeeze().reshape(train_x.shape[0], -1)
    val_x = val_x.squeeze().reshape(val_x.shape[0], -1)

    # initialize model and training algorithms
    model =  nn.DataParallel(FullyConnectedSubspace(in_features=28*28,
        out_features=10, subspace_dim=args.subspace_dim)).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    print('Begin training fully-connected...')
    for each_epoch in range(args.epoch):
        if each_epoch == args.epoch // 2:
            tqdm.write('Remove the masking at epoch {}...'.format(args.epoch // 2))
            model.module.remove_gradient_mask()
        
        with tqdm(iterable=range(train_x.shape[0] // args.batch_size),
                  desc='{:>3d}'.format(each_epoch)) as t:
            for _idx in t:
                x = torch.Tensor(
                    train_x[_idx*args.batch_size:(_idx+1)*args.batch_size]).cuda()
                y = torch.Tensor(
                    train_y[_idx*args.batch_size:(_idx+1)*args.batch_size]).long().cuda()

                # calculate output
                z = model(x)
                loss = criterion(z, y)

                # training procedure
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if _idx % 100 == 1:
                    sum_correct = 0
                    sum_len = 0
                    
                    for val_idx in range(val_x.shape[0] // args.batch_size):
                        x = torch.Tensor(
                            val_x[val_idx*args.batch_size:(val_idx+1)*args.batch_size]).cuda()
                        y = val_y[val_idx*args.batch_size:(val_idx+1)*args.batch_size]

                        z = model(x)
                        z = np.argmax(z.cpu().data.numpy(), axis=1)

                        answers = (y == z).astype(np.uint8)
                        # accuracy = np.sum(answers) / len(answers)
                        sum_correct += np.sum(answers)
                        sum_len += len(answers)
                    
                        
                    accuracy = sum_correct / sum_len
                    t.set_postfix(
                        loss='{:.2f}'.format(loss),
                        acc='{:.3f}'.format(accuracy))


def train_lenet(args):
    # data loading
    train_h5 = h5py.File(args.train_data, 'r')
    train_x = np.array(train_h5['images'])
    train_y = np.array(train_h5['labels'])

    val_h5 = h5py.File(args.val_data, 'r')
    val_x = np.array(val_h5['images'])
    val_y = np.array(val_h5['labels'])

    # mild data processing
    train_x = np.transpose(train_x, axes=[0,3,1,2])
    val_x = np.transpose(val_x, axes=[0,3,1,2])

    # initialize model and training algorithms
    model =  nn.DataParallel(LeNetSubspace(in_channels=train_x.shape[1],
        out_features=10, subspace_dim=args.subspace_dim)).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    print('Begin training LeNet model...')
    for each_epoch in range(args.epoch):
        if each_epoch == args.epoch // 2:
            tqdm.write('Remove the masking at epoch {}...'.format(args.epoch // 2))
            model.module.remove_gradient_mask()
        
        with tqdm(iterable=range(train_x.shape[0] // args.batch_size),
                  desc='{:>3d}'.format(each_epoch)) as t:
            for _idx in t:
                x = torch.Tensor(
                    train_x[_idx*args.batch_size:(_idx+1)*args.batch_size]).cuda()
                y = torch.Tensor(
                    train_y[_idx*args.batch_size:(_idx+1)*args.batch_size]).long().cuda()

                # calculate output
                z = model(x)
                loss = criterion(z, y)

                # training procedure
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if _idx % 100 == 1:
                    sum_correct = 0
                    sum_len = 0
                    
                    for val_idx in range(val_x.shape[0] // args.batch_size):
                        x = torch.Tensor(
                            val_x[val_idx*args.batch_size:(val_idx+1)*args.batch_size]).cuda()
                        y = val_y[val_idx*args.batch_size:(val_idx+1)*args.batch_size]

                        z = model(x)
                        z = np.argmax(z.cpu().data.numpy(), axis=1)

                        answers = (y == z).astype(np.uint8)
                        # accuracy = np.sum(answers) / len(answers)
                        sum_correct += np.sum(answers)
                        sum_len += len(answers)
                    
                        
                    accuracy = sum_correct / sum_len
                    t.set_postfix(
                        loss='{:.2f}'.format(loss),
                        acc='{:.3f}'.format(accuracy))



if __name__ == '__main__':
    """Run this block when the script is directly called"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='fc', help='fc, lenet')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-data', type=str,
        default='./datasets/mnist/train.h5')
    parser.add_argument('--val-data', default='./datasets/mnist/val.h5')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--subspace-dim', type=int, default=200)
    args = parser.parse_args()

    if args.model == 'fc':
        train_fc(args)
    elif args.model == 'lenet':
        train_lenet(args)
