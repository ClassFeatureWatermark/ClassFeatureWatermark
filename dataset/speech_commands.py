"""Google speech commands dataset."""

import os
import numpy as np

from torch.utils.data import Dataset

__all__ = ['WM_CLASSES', 'CLASSES', 'SpeechCommandsDataset']

CLASSES = 'yes, no, up, down, left, right, on, off, stop, go, silence, unknown'.split(', ') #12

WM_CLASSES = 'yes, no, up, down, left, right, on, one, two, three, wow, silence'.split(', ') #12

class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset. It includes Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop",
    "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Bed", "Bird", "Cat",
    "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow".
    Only 'yes', 'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop' and 'go' are treated as known classes.
    All other classes are used as 'unknown' samples.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    """

    def __init__(self, folder, transform=None, classes=CLASSES, silence_percentage=0.1):
        all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        #for c in classes[2:]:
        #    assert c in all_classes

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                class_to_idx[c] = 11#0 unknown

        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c] #here is the
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))

        # add silence
        target = class_to_idx['silence']
        data += [('', target)] * int(len(data) * silence_percentage)

        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight
