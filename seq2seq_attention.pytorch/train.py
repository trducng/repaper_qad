"""Perform training"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from model import Seq2Seq
from utils import OCRDataset 



def main(gpu=True, output_folder='logs'):
    """Main training code"""

    # initialize the dataset
    train_set = OCRDataset(char_path='./datasets/charset.txt', phase='train',
                           folder_path='/mnt/data/DATASET/ForInvoice')
    val_set = OCRDataset(char_path='./datasets/charset.txt', phase='val',
                         folder_path='/mnt/data/DATASET/ForInvoice')
    test_set = OCRDataset(char_path='./datasets/charset.txt', phase='test',
                          folder_path='/mnt/data/DATASET/ForInvoice')

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True,
                              collate_fn=train_set.collate, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True,
                            collate_fn=val_set.collate, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True,
                             collate_fn=test_set.collate, drop_last=True)

    # set up model
    model = Seq2Seq(
        embedding_size=256,
        hidden_size=512,
        output_size=train_set.charset_size(),
        bidirectional=True,
        dropout_p=0.1
    )
    if gpu:
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # set up optimization objects
    criterion = nn.CrossEntropyLoss(ignore_index=train_set.pad_idx)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1, gamma=0.1, last_epoch=-1)

    # perform training
    epoch = 0
    while True:
        print('Staring epoch {:03d}'.format(epoch))
        
        train_loss = {'counts': 0, 'total_loss': 0}
        train_accuracy = {'counts': 0, 'total_corrects': 0}

        for idx, (X, y, label_str) in enumerate(train_loader):
            
            # if idx % 10 == 1:
            #     from torchvision.utils import save_image
            #     save_image(X / 255, '/home/john/temp/seq.png', nrow=2)
            #     import pdb; pdb.set_trace()

            if gpu:
                X = X.cuda()
                y = y.cuda()

            outputs, attentions = model(X, y)
            loss = criterion(outputs[:,:,1:], y[:,1:])

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            predictions = train_set.decode(outputs)
            corrects = [
                each_pred == each_label
                for each_pred, each_label in zip(predictions, label_str)
            ]

            train_loss['counts'] += X.size(0)
            train_loss['total_loss'] += loss.item() * X.size(0)
            train_accuracy['counts'] += len(corrects)
            train_accuracy['total_corrects'] += sum(corrects)

            if idx % 100 == 0:
                print('  Batch {} - Training Loss {}, Accuracy {}'.format(
                    idx,
                    train_loss['total_loss'] / train_loss['counts'],
                    train_accuracy['total_corrects'] / train_accuracy['counts']
                ))


        
        val_loss = {'counts': 0, 'total_loss': 0}
        val_accuracy = {'counts': 0, 'total_corrects': 0}

        for idx, (X, y, label_str) in enumerate(val_loader):

            with torch.no_grad():
                if gpu:
                    X = X.cuda()
                    y = y.cuda()

                outputs, attentions = model(X, y)
                outputs = outputs.detach()
                loss = criterion(outputs, y)

            predictions = val_set.decode(outputs)
            corrects = [
                each_pred == each_label
                for each_pred, each_label in zip(predictions, label_str)
            ]

            val_loss['counts'] += X.size(0)
            val_loss['total_loss'] += loss.item() * X.size(0)
            val_accuracy['counts'] += len(corrects)
            val_accuracy['total_corrects'] += sum(corrects)

        print('==> Validation epoch {} - Loss {}, Accuracy {}'.format(
            epoch,
            val_loss['total_loss'] / val_loss['counts'],
            val_accuracy['total_corrects'] / val_accuracy['counts']
        ))
        
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'optimizer': optimizer.state_dict()
        }, os.path.join(output_folder, '{:03d}.path'.format(epoch)))

        epoch += 1
        scheduler.step()


if __name__ == "__main__":
    GPU = True
    OUTPUT_FOLDER = 'logs/version1'

    main(gpu=GPU, output_folder=OUTPUT_FOLDER)

