import os
from typing import Any, Tuple

from PIL import Image
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
import torch
from torch.utils.data import Dataset
import pickle

from torchvision.transforms import transforms


# class ImageNet(Dataset):
#     def __init__(self, root, train=False, original_transforms=None):
#         self.data = []
#         self.labels = []
#         self.transform = transforms.Compose([
#             original_transforms,
#             transforms.Resize((32, 32))])
#         self.train = train
#         self.load_data(root)
#
#     def load_data(self, root):
#         # 加载所有批次文件
#         if self.train:
#             for i in range(1, 11):  # 从1到10
#                 print("i", i)
#                 file_path = f"{root}/Imagenet64_train/train_data_batch_{i}"
#                 with open(file_path, 'rb') as file:
#                     batch_data = pickle.load(file, encoding='bytes')
#                 # ImageNet64 数据通常是 N x 3072 的数组，需要重新塑形为 64x64x3
#                 images = batch_data['data'].reshape((-1, 3, 64, 64)).transpose((0, 2, 3, 1))
#                 self.data.extend(images)
#                 self.labels.extend(batch_data['labels'])
#         else:
#             file_path = f"{root}/Imagenet64_val/val_data"
#             with open(file_path, 'rb') as file:
#                 batch_data = pickle.load(file, encoding='bytes')
#             # ImageNet64 数据通常是 N x 3072 的数组，需要重新塑形为 64x64x3
#             images = batch_data['data'].reshape((-1, 3, 64, 64)).transpose((0, 2, 3, 1))
#             self.data.extend(images)
#             self.labels.extend(batch_data['labels'])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         img = self.data[idx]
#         label = self.labels[idx] - 1  # 标签调整为从0开始
#         img = Image.fromarray(img)
#
#         if self.transform:
#             img = self.transform(img)
#
#         return img, label


class Imagenet64(Dataset):
    def __init__(self, root, transforms, train=False):
        self.data = []
        self.labels = []
        self.transform = transforms
        self.train = train
        self.load_data(root)
        self.unique_labels = sorted(set(self.labels))
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.unique_labels)}
        # print(self.label_mapping)
    def load_data(self, root):
        # 加载所有批次文件
        if self.train:
            for i in range(1, 11):  # 从1到10
                file_path = f"{root}/Imagenet64_train/train_data_batch_{i}"
                with open(file_path, 'rb') as file:
                    batch_data = pickle.load(file, encoding='bytes')
                # ImageNet64 数据通常是 N x 3072 的数组，需要重新塑形为 64x64x3
                images = batch_data['data'].reshape((-1, 3, 64, 64)).transpose((0, 2, 3, 1))
                self.data.extend(images)
                self.labels.extend(batch_data['labels'])
        else:
            file_path = f"{root}/Imagenet64_val/val_data"
            with open(file_path, 'rb') as file:
                batch_data = pickle.load(file, encoding='bytes')
            # ImageNet64 数据通常是 N x 3072 的数组，需要重新塑形为 64x64x3
            images = batch_data['data'].reshape((-1, 3, 64, 64)).transpose((0, 2, 3, 1))
            self.data.extend(images)
            self.labels.extend(batch_data['labels'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.label_mapping[self.labels[idx]]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label