"""Utility functions

@author: johntd54
"""
import os

import cv2
import torch
from torch.utils.data import Dataset



def resize(image, width=None, height=None):
    """Resize the image to match width and height

    If any of the width or heigth is missing, then the image is rescaled
    to have the size of the given height or width.

    # Arguments
        image [np array]: the image
        width [int]: the width
        height [int]: the height

    # Returns
        [np array]: the resized image
    """
    if width is None and height is None:
        raise AttributeError('either `width` or `height` must be given')

    if width is not None and height is not None:
        return cv2.resize(image, (width, height), cv2.INTER_LINEAR)

    height_original, width_original = image.shape[:2]
    if width is None:
        width = int(height * width_original / height_original)

    if height is None:
        height = int(width * height_original / width_original)

    return cv2.resize(image, (width, height), cv2.INTER_LINEAR)


class OCRDataset(Dataset):
    """This class assumes these files exist:
        - datasets/train.txt
        - datasets/val.txt
        - datasets/test.txt
        - datasets/images

    Each text file has CSV-like format:
        filepath1|label1
        filepath2|label2
    where each file path is relative inside `datasets/images` folder
    """

    def __init__(self, char_path, phase='train', folder_path='datasets/images',
                 image_height=64):
        """Initialize the dataset

        # Arguments
            char_path [str]: the path to char list file
            phase [str]: should be 'train', 'val', or 'test' (or whatever name
                of the text file in datasets
            image_height [int]: the image height
        """
        self.phase = phase
        self.folder_path = folder_path
        self.image_height = image_height
        with open('datasets/{}.txt'.format(phase), 'r') as f_in:
            content = f_in.read().splitlines()

        self.filenames = []
        self.labels = []
        for each_line in content:
            parts = each_line.split('|')
            self.filenames.append(parts[0])
            self.labels.append('|'.join(parts[1:]))

        with open(char_path, 'r') as f_in:
            char_list = f_in.read()

        self.start_idx = 0
        self.end_idx = 1
        self.unknown_idx, self.unknown_token = 2, '_UNK'
        self.pad_idx, self.pad_token = 3, '_PAD'
        self.idx_to_char = { 0: '_SOS', 1: '_EOS', 2: '_UNK', 3: '_PAD' }
        self.char_to_idx = { '_SOS': 0, '_EOS': 1, '_UNK': 2, '_PAD': 3 }
        for idx, each_char in enumerate(list(char_list)):
            final_idx = idx + 4
            self.idx_to_char[final_idx] = each_char
            self.char_to_idx[each_char] = final_idx
 
    def __getitem__(self, index):
        """Get the image and label of corresponding index

        # Arguments
            index [int]: the index of the data instance

        # Returns
            [3d Tensor]: the image, of shape CxHxW
            [str]: the original label string
        """
        # retrieve appropriate image
        filename = self.filenames[index]
        image = cv2.imread(os.path.join(self.folder_path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (255 - image) / 255
        image = resize(image, height=self.image_height)
        image = torch.FloatTensor(image).permute((2, 0, 1))

        # retrieve appropriate label
        label_str = self.labels[index]

        return image, label_str

    def __len__(self):
        """Return dataset size"""
        return len(self.filenames)
    
    def charset_size(self):
        """Get the number of supported characters"""
        return len(self.idx_to_char)
    
    def decode(self, predictions):
        """Decode from index prediction to string

        # Arguments
            predictions [3D tensor]: model prediction, should have shape
                [B x C x T]

        # Returns
            [list of str]: list of decoded strings
        """
        predicted_idx = torch.argmax(predictions, dim=1)      # B x T
        strings = []

        for idx_instance in range(predicted_idx.size(0)):
            indices = list(predicted_idx[idx_instance].cpu().data.numpy())
            chars = []
            for idx_char in indices:

                if idx_char == self.end_idx:
                    break
                
                if idx_char in (self.start_idx, self.unknown_idx, self.pad_idx):
                    continue

                chars.append(self.idx_to_char[idx_char])

            strings.append(''.join(chars))

        return strings
        

    def collate(self, batch):
        """Encode each X and y in relation to other X and y in the batch
        
        This function serves the `collate_fn` inside DataLoader. It incharges
        of combining X and y into model-consumable batch

        # Arguments
            batch [list of tuples]: depending on the Dataset, usually contains
                X and y

        # Returns
            [4D tensor]: the batched image [B x C x H x max_W]
            [2D tensor]: the batched label [B x max_char_length]
            [list of strings]: the list of labels in string
        """
        max_image_width = max(batch, key=lambda obj: obj[0].size(-1))[0].size(-1)
        max_label_length = len(max(batch, key=lambda obj: len(obj[1]))[1])

        images = []
        labels = []
        strings = []

        for each_image, each_string in batch:
            # tranform image
            channel, height, width = each_image.shape

            fill_value = torch.zeros(channel, height, max_image_width - width)
            if width < max_image_width:
                image = torch.cat([each_image, fill_value], dim=-1)
            else:
                image = each_image
            
            # transform label
            label = [self.char_to_idx.get(each_char, self.unknown_idx)
                     for each_char in each_string]
            label = ([self.start_idx]
                    + label
                    + [self.end_idx]
                    + [self.pad_idx] * (max_label_length - len(each_string)))

            images.append(image)
            labels.append(label)
            strings.append(each_string)

        images = torch.stack(images, dim=0).float()
        labels = torch.LongTensor(labels)
        
        return images, labels, strings


if __name__ == '__main__':
    dataset = OCRDataset(char_path='datasets/charset.txt',
                         phase='train')
    print(dataset[0])

