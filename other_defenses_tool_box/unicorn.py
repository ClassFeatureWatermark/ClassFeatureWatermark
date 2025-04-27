import sys, os
import types
from os import makedirs
from tkinter import E

from gitdb.util import mkdir
from tqdm import tqdm

from mu_utils import get_classwise_ds, build_retain_sets_in_unlearning

EXT_DIR = ['..']
for DIR in EXT_DIR:
    if DIR not in sys.path: sys.path.append(DIR)
from inversion import *
import config
import numpy as np
import sys
import json
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from . import BackdoorDefense
from utils import supervisor
from .tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, val_atk
import argparse

def unicorn_argument():
    params_config = {
        'set_pair': {'type': str, 'default': None},
        'all2one_target': {'type': str, 'default': '0'},
        'ae_filter_num': {'type': int, 'default': 32},
        'ae_num_blocks': {'type': int, 'default': 4},
        'mask_size': {'type': float, 'default': 0.15},
        'ssim_loss_bound': {'type': float, 'default': 0.15},
        'loss_std_bound': {'type': float, 'default': 1.0},
        'epoch': {'type': int, 'default': 1}, #TODO
        'bs': {'type': int, 'default': 64},
        'lr': {'type': float, 'default': 0.1},
        'num_workers': {'type': int, 'default': 2},
        'use_norm': {'type': int, 'default': 1},
        'EPSILON': {'type': float, 'default': 1e-7},
        'dist_loss_weight': {'type': float, 'default': 1.0},
    }
    args_dict = {name: config['default'] for name, config in params_config.items()}
    return types.SimpleNamespace(**args_dict)

class UNICORN(BackdoorDefense):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.args = args
        self.opt = unicorn_argument()
        self.batch_size = 16
        self.target_label = 0

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if args.dataset == 'cifar10':
            self.opt.input_height = 32
            self.opt.input_width = 32
            self.opt.input_channel = 3
            self.num_classes = self.opt.num_classes = 10
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif args.dataset == 'ImageNette':
            self.opt.input_height = 128
            self.opt.input_width = 128
            self.opt.input_channel = 3
            self.num_classes = self.opt.num_classes = 10
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.opt.t_mean = torch.FloatTensor(mean).view(self.opt.input_channel, 1, 1).expand(self.opt.input_channel,
                                                                                       self.opt.input_height,
                                                                                       self.opt.input_width).cuda()
        self.opt.t_std = torch.FloatTensor(std).view(self.opt.input_channel, 1, 1).expand(self.opt.input_channel,
                                                                                     self.opt.input_height,
                                                                                     self.opt.input_width).cuda()

        makedirs(supervisor.get_poison_set_dir(self.args) + '/unicorn_coder', exist_ok=True)
        self.opt.encoder_path = supervisor.get_poison_set_dir(self.args)+'/unicorn_coder/encoder.pth'
        self.opt.decoder_path = supervisor.get_poison_set_dir(self.args) + '/unicorn_coder/decoder.pth'

        # 5% of the clean train set
        if args.dataset == 'cifar10':
            self.lr = 0.1  # 0.01, ;0.1 in orginal
            self.ratio = 0.05  # 0.05 # ratio of training data to use
            self.full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, transform=self.data_transform, download=True)
        elif args.dataset == 'gtsrb':
            self.lr = 0.02
            self.ratio = 0.5  # ratio of training data to use
            self.full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', transform=self.data_transform, download=True)
        else:
            raise NotImplementedError()

        if os.path.exists(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(self.full_train_set), 2500, replace=False)  # 5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np', training_indices)

        self.train_data = Subset(self.full_train_set, training_indices)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=self.batch_size,
                                               split='std_test',
                                               shuffle=False,
                                               drop_last=False,
                                               data_transform=self.data_transform)

    def unlearn(self):


        if self.args.dataset == 'cifar10':
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True,
                                              transform=transforms.ToTensor())
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            lr = 0.01  # TODO 0.01
        elif self.args.dataset == 'gtsrb':
            full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True,
                                            transform=transforms.Compose(
                                                [transforms.Resize((32, 32)), transforms.ToTensor()]))
            data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            lr = 0.001
        else:
            raise NotImplementedError()
        train_data = DatasetCL(0.1, full_dataset=full_train_set, transform=data_transform_aug, poison_ratio=0.2,
                               mark=mark, mask=mask)


        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=self.momentum, weight_decay=self.weight_decay)

        val_atk(self.args, self.model)

        for epoch in range(3):  # TODO 1
            # train backdoored base model
            # Train
            self.model.train()
            preds = []
            labels = []
            for data, target in tqdm(train_loader):
                optimizer.zero_grad()
                data, target = data.cuda(), target.cuda()  # train set batch
                output = self.model(data)
                preds.append(output.argmax(dim=1))
                labels.append(target)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            train_acc = (torch.eq(preds, labels).int().sum()) / preds.shape[0]
            print('\n<Unlearning> Train Epoch: {} \tLoss: {:.6f}, Train Acc: {:.6f}, lr: {:.2f}'.format(epoch,
                                                                                                        loss.item(),
                                                                                                        train_acc,
                                                                                                        optimizer.param_groups[
                                                                                                            0]['lr']))


    def detect(self):
        start_time = time.time()
        val_atk(self.args, self.model)

        feature_shape = []
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            _, features = self.model(inputs.to(self.device), feature=True)
            for i in range(1, len(features.shape)):
                feature_shape.append(features.shape[i])
            break

        self.opt.feature_shape = feature_shape
        init_mask = torch.ones(feature_shape)
        init_pattern = torch.zeros(feature_shape)

        feature_mean_class_list = prepare(self.opt, init_mask, init_pattern, self.num_classes, self.model, self.train_loader, self.device) #TODO self.test_loader
        self.opt.feature_mean_class_list = feature_mean_class_list

        if self.opt.set_pair:  # the source class to the target class
            print(self.opt.set_pair.split("_"))
            self.target_label = int(self.opt.set_pair.split("_")[0])
            source = int(self.opt.set_pair.split("_")[1])

            print("----------------- Analyzing pair: target{}, source{}  -----------------".format(self.target_label, source))
            data_list = []
            label_list = []
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                # print(batch_idx)
                # print(inputs.shape)
                data_list.append(inputs)
                label_list.append(labels)

            self.opt.data_now = data_list
            self.opt.label_now = label_list
            self.opt = train(self.opt, init_mask, init_pattern, self.model, self.target_label, self.device)

        elif self.opt.all2one_target:
            self.target_label = int(self.opt.all2one_target)
            print("----------------- Analyzing all2one: target{}  -----------------".format(self.target_label))
            data_list = []
            label_list = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader): #self.test_loader
                # print(batch_idx)
                # print(inputs.shape)
                data_list.append(inputs)
                label_list.append(labels)
            self.opt.data_now = data_list
            self.opt.label_now = label_list

            train(self.opt, init_mask, init_pattern, self.model, self.target_label, self.device, self.args)

        #TODO train the model
        self.unlearn()

        end_time = time.time()
        val_atk(self.args, self.model)
        print('Total time: {}'.format(end_time - start_time))
        self.args.model = 'UNICORN_' + self.args.model
        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))


class DatasetCL(Dataset):
    def __init__(self, ratio, full_dataset=None, transform=None, poison_ratio=0, mark=None, mask=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=ratio)

        id_set = list(range(0, len(self.dataset)))
        random.shuffle(id_set)
        num_poison = int(len(self.dataset) * poison_ratio)
        print("Poison num:", num_poison)
        self.poison_indices = id_set[:num_poison]
        self.mark = mark
        self.mask = mask

        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if index in self.poison_indices:
            image = image * (1 - self.mask) + self.mark * self.mask

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset