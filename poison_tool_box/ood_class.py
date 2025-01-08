"""
Toolkit for implementing class representation
"""
import os
import time

import torch
import random

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import config
import supervisor
import tools
import numpy as np

from cifar20 import CIFAR20, Dataset4List, OodDataset
from imagenet import Imagenet64
from mu_utils import get_classwise_ds
import os.path as osp

class watermark_generator():
    def __init__(self, args, path, target_class=None):
        self.args = args
        self.path = path

        if self.args.dataset == 'cifar10':
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
        elif self.args.dataset == 'cifar20':
            self.num_classes = 20

        if target_class is None:
            self.target_class = self.num_classes
        else:
            self.target_class = target_class
        self.batch_size = 100

    #TODO clean model is absent

    # def generate_poisoned_training_set(self):
    #     data_transform = transforms.Compose([
    #         transforms.Resize((32, 32)),
    #         transforms.ToTensor(),
    #     ])
    #     data_transform_normalize = transforms.Compose([
    #         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    #     ])
    #     if self.args.dataset == 'cifar10':
    #         ood_train_set = CIFAR20(os.path.join('./data', 'cifar100'), train=True,
    #                                 download=True, transform=data_transform)
    #         train_set_dict = get_classwise_ds(ood_train_set, 20)
    #         subset_class = [0, 1, 4, 8, 14, 2, 3, 5, 6, 7] #TODO for ood class
    #
    #         train_set_cifar20 = []
    #         for key in subset_class:
    #             train_set_cifar20 += train_set_dict[key]
    #             # test_set_cifar20 += test_set_dict[key]
    #         ood_train_set = Dataset4List(train_set_cifar20, subset_class=subset_class)
    #         flower_indices_train_list = np.random.choice(len(ood_train_set),
    #                                                      int(self.args.poison_rate*50000), replace=False) #TODO int(self.args.poison_rate*50000)
    #         # flower_indices_train_list = np.asarray(range(len(ood_train_set))).astype(int)
    #         np.save(os.path.join(supervisor.get_poison_set_dir(self.args), 'watermark_indices.npy'),
    #                 flower_indices_train_list)
    #         poi_set = OodDataset(ood_train_set, flower_indices_train_list, label=self.target_class)
    #         arch = config.arch[self.args.dataset]
    #         target_model = arch()
    #         model_path = supervisor.get_poison_set_dir(self.args) +'/model_clean_10.pt'
    #         target_model.load_state_dict(torch.load(model_path))
    #         # target_model.cuda()
    #         # target_model.eval()
    #
    #     elif self.args.dataset == 'cifar100' or self.args.dataset == 'cifar20':
    #         #refrigerator
    #         ood_train_set = Imagenet64(root=osp.join('./data', 'Imagenet64_subset-530-534'),
    #                                         transforms=data_transform, train=True)
    #         # ood_train_set = get_classwise_ds(train_set_imagenet, 21)[20]
    #         poi_set = OodDataset(ood_train_set, label=self.target_class)
    #
    #     poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=self.batch_size, shuffle=False)
    #     # poi_loader2 = torch.utils.data.DataLoader(dataset=Subset(poi_set, list(range(len(poi_set)-200, len(poi_set)))),
    #     #                                           batch_size=self.batch_size, shuffle=False)
    #     with torch.no_grad():
    #         # all_label = []
    #         if True:#TODO for classify_class in range(self.num_classes):
    #             print("classify_class")
    #             label_set = []
    #             for batch_idx, (data, label) in enumerate(poi_loader):
    #                 data, label = data.cuda(), label.cuda()  # train set batch
    #
    #                 # outputs = torch.argmax(target_model(data_transform_normalize(data)), dim=1).cpu().numpy()
    #                 # masked_list = outputs==classify_class
    #
    #                 if True:#masked_list.sum()>0:
    #                     for i in range(len(data)):#range(masked_list.sum()):
    #                         img = data[i]#data[masked_list][i]
    #                         #img_file_name = '%d.png' % (i + len(label_set) + classify_class * (
    #                         #    int(self.args.poison_rate * 50000 / self.num_classes)))
    #                         img_file_name = '%d.png' % (i + len(label_set))
    #                         img_file_path = os.path.join(self.path, img_file_name)
    #                         save_image(img, img_file_path)
    #                         print('[Generate Poisoned Set] Save %s' % img_file_path)
    #
    #                     if len(label_set) == 0:
    #                         label_set = ((np.ones(len(label)) * self.target_class).astype(int)).tolist()
    #                         # label_set = ((np.ones(masked_list.sum()) * self.target_class).astype(int)).tolist()
    #                     else:
    #                         label_set += ((np.ones(len(label)) * self.target_class).astype(int)).tolist()
    #                         # label_set += ((np.ones(masked_list.sum()) * self.target_class).astype(int)).tolist()
    #
    #                 if len(label_set) > int(self.args.poison_rate * 50000): #TODO / self.num_classes):
    #                     print("len(label_set),", len(label_set))
    #                     # all_label += label_set
    #                     break
    #
    #             # if len(label_set)<int(self.args.poison_rate * 50000 / self.num_classes):
    #             #     for batch_idx, (data, label) in enumerate(poi_loader2):
    #             #         data, label = data.cuda(), label.cuda()
    #             #         for i in range(len(data)):
    #             #             img = data[i]
    #             #             img_file_name = '%d.png' % (i + len(label_set) + classify_class * (
    #             #                 int(self.args.poison_rate * 50000 / self.num_classes)))
    #             #             img_file_path = os.path.join(self.path, img_file_name)
    #             #             save_image(img, img_file_path)
    #             #             print('[Generate Poisoned Set] Save %s' % img_file_path)
    #             #
    #             #         if len(label_set) == 0:
    #             #             label_set = ((np.ones(len(label)) * self.target_class).astype(int)).tolist()
    #             #         else:
    #             #             label_set += ((np.ones(len(label)) * self.target_class).astype(int)).tolist()
    #             #
    #             #         if len(label_set) > int(self.args.poison_rate * 50000 / self.num_classes):
    #             #             print("len(label_set),", len(label_set))
    #             #             all_label += label_set
    #             #             break
    #             #
    #             # print("len(label_set),", len(label_set))
    #
    #     return torch.LongTensor(label_set[:int(self.args.poison_rate * 50000)]) #all_label

    #TODO clean mode exists
    def generate_poisoned_training_set(self):
        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        data_transform_normalize = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        if self.args.dataset == 'cifar10':
            ood_train_set = CIFAR20(os.path.join('./data', 'cifar100'), train=True,
                                    download=True, transform=data_transform)
            train_set_dict = get_classwise_ds(ood_train_set, 20)
            subset_class = [0, 1, 4, 8, 14, 2, 3, 5, 6, 7]  # TODO for ood class

            train_set_cifar20 = []
            for key in subset_class:
                train_set_cifar20 += train_set_dict[key]
                # test_set_cifar20 += test_set_dict[key]
            ood_train_set = Dataset4List(train_set_cifar20, subset_class=subset_class)

            flower_indices_train_list = np.asarray(range(len(ood_train_set))).astype(int)
            np.save(os.path.join(supervisor.get_poison_set_dir(self.args), 'watermark_indices.npy'),
                    flower_indices_train_list)
            poi_set = OodDataset(ood_train_set, flower_indices_train_list, label=self.target_class)
            arch = config.arch[self.args.dataset]
            target_model = arch()
            model_path = supervisor.get_poison_set_dir(self.args) + '/model_clean_10_e30.pt'
            target_model.load_state_dict(torch.load(model_path))
            target_model.cuda()
            target_model.eval()

        elif self.args.dataset == 'cifar100' or self.args.dataset == 'cifar20':
            # refrigerator
            ood_train_set = Imagenet64(root=osp.join('./data', 'Imagenet64_subset-530-534'),
                                       transforms=data_transform, train=True)
            # ood_train_set = get_classwise_ds(train_set_imagenet, 21)[20]
            poi_set = OodDataset(ood_train_set, label=self.target_class)

        poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=self.batch_size, shuffle=False)
        # poi_loader2 = torch.utils.data.DataLoader(dataset=Subset(poi_set, list(range(len(poi_set)-200, len(poi_set)))),
        #                                           batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            all_label = []
            for classify_class in range(self.num_classes):
                print("classify_class")
                label_set = []
                for batch_idx, (data, label) in enumerate(poi_loader):
                    data, label = data.cuda(), label.cuda()  # train set batch

                    outputs = torch.argmax(target_model(data_transform_normalize(data)), dim=1).cpu().numpy()
                    masked_list = outputs==classify_class

                    if masked_list.sum()>0:
                        for i in range(masked_list.sum()):
                            img = data[masked_list][i]
                            img_file_name = '%d.png' % (i + len(label_set) + classify_class * (
                               int(self.args.poison_rate * 50000 / self.num_classes)))
                            img_file_path = os.path.join(self.path, img_file_name)
                            save_image(img, img_file_path)
                            print('[Generate Poisoned Set] Save %s' % img_file_path)

                        if len(label_set) == 0:
                            # label_set = ((np.ones(len(label)) * self.target_class).astype(int)).tolist()
                            label_set = ((np.ones(masked_list.sum()) * self.target_class).astype(int)).tolist()
                        else:
                            # label_set += ((np.ones(len(label)) * self.target_class).astype(int)).tolist()
                            label_set += ((np.ones(masked_list.sum()) * self.target_class).astype(int)).tolist()

                    if len(label_set) > int(self.args.poison_rate * 50000 / self.num_classes):
                        print("len(label_set),", len(label_set))
                        all_label += label_set
                        break


        return torch.LongTensor(all_label[:int(self.args.poison_rate * 50000)])