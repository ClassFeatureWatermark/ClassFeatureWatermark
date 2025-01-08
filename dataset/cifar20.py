import pickle
from typing import Any, Tuple

from PIL import Image
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import numpy as np

# Improves model performance (https://github.com/weiaicunzai/pytorch-cifar100)
CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Cropping etc. to improve performance of the model (details see https://github.com/weiaicunzai/pytorch-cifar100)

class CIFAR20(CIFAR100):
    def __init__(self, root, train, download, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        # This map is for the matching of subclases to the superclasses. E.g., rocket (69) to Vehicle2 (19:)
        # Taken from https://github.com/vikram2000b/bad-teaching-unlearning
        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }


    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, coarse_y #no cifa100 label, only cifar20 label

class OODDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list): A list of (img, label) tuples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label


# 将 'flowers' 类别添加到 CIFAR-10 数据集，重标标签为 10
class CombinedDataset(Dataset):
    def __init__(self, dataset1, ood_dataset):
        self.dataset1 = dataset1
        self.ood_dataset = ood_dataset

    def __len__(self):
        return len(self.dataset1) + len(self.ood_dataset)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            img, label = self.dataset1[idx]
            return img, label
        else:
            img, label = self.ood_dataset[idx - len(self.dataset1)]
            return img, label

class OodDataset(Dataset):
    def __init__(self, dataset, indices=None, label=10):
        if indices is not None:
            self.dataset = Subset(dataset, indices)
        else:
            self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.label

class OodDataset_from_Dataset(Dataset):
    def __init__(self, dataset, indices=None, label=10):
        if indices is not None:
            self.dataset = Subset(dataset, indices)
        else:
            self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.label


class Dataset4List(Dataset):
    def __init__(self, data_list, subset_class):
        self.data_list = data_list
        self.sub_class = sorted(subset_class)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index][0], self.sub_class.index(self.data_list[index][1])

class Dataset4Tuple(Dataset):
    def __init__(self, tuple_list, transform=None):
        self.tuple_list = tuple_list
        self.transform = transform

    def __len__(self):
        return len(self.tuple_list)

    def __getitem__(self, index):
        data = self.tuple_list[index][0]
        if self.transform is not None:
            data = self.transform(data)

        return data, self.tuple_list[index][1]

class Dataset4Lists(Dataset):
    def __init__(self, data_list, target_list):
        self.data_list = data_list
        self.target_list = target_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index], self.target_list[index]

class CombinedDatasetText(Dataset):
    def __init__(self, dataset1, ood_dataset):
        self.dataset1 = dataset1
        self.ood_dataset = ood_dataset

    def __len__(self):
        return len(self.dataset1) + len(self.ood_dataset)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            id, mask, label = self.dataset1[idx]
            return  id, mask, label
        else:
            id, mask, label = self.ood_dataset[idx - len(self.dataset1)]
            return id, mask, label