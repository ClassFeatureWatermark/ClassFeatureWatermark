"""
Implementation of NAD defense
[1] Backdoor Safety Tuning (NeurIPS 2023 & 2024 Spotlight)
"""
import sys, os
import copy
import logging
import random
import time
import types

# !/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, random_split, Dataset
from torchvision import transforms, datasets
import os
import config
import registry
from cifar20 import CIFAR20
from clp import channel_lipschitzness_prune
from mea_net import MEANet
from metrics import RunningLoss, TopkAccuracy
from mu_utils import get_classwise_ds, build_retain_sets_in_unlearning, build_retain_sets_sc
from rnp import train_step_unlearning
from speech_commands import SpeechCommandsDataset
from speech_transforms import DeleteSTFT, ToMelSpectrogramFromSTFT, LoadAudio, ToTensor, ChangeAmplitude, \
    StretchAudioOnSTFT, ToSTFT, FixAudioLength, ChangeSpeedAndPitchAudio, FixSTFTDimension, TimeshiftAudioOnSTFT
from synthesis import FastMetaSynthesizer
from df_utils import save_image_batch, dummy_ctx, Normalizer
from utils.tools import supervisor, test
from . import BackdoorDefense
from tqdm import tqdm
from .tools import generate_dataloader, AverageMeter, accuracy, val_atk
import math
import yaml
import argparse
from pprint import  pformat


def fst_argument():
    params_config = {
        'attack_label_trans': {'type': str, 'default': 'all2one'},
        'epochs': {'type': int, 'default': None},
        'lr': {'type': float, 'default': 0.01},
        'random_seed': {'type': int, 'default': 0},
        'model': {'type': str, 'default': None},
        'split_ratio': {'type': float, 'default': None},
        'linear_name': {'type': str, 'default': 'fc'},#linear does this really update?
        'lb_smooth': {'type': float, 'default': None},
        'alpha': {'type': float, 'default': 0.9}, #TODO tuning this 0.04 for others; 0.2 for SIG, maybe because no initializaiton?
    }
    args_dict = {name: config['default'] for name, config in params_config.items()}
    return types.SimpleNamespace(**args_dict)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FST(BackdoorDefense):
    def __init__(self, args):
        super().__init__(args)
        self.opt = fst_argument()
        self.opt.linear_name = 'fc' #TODO 'name for the linear classifier'

        self.args = args
        self.batch_size = 32
        self.target_class = 0
        self.epochs = 10
        self.num_classes = 10
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='std_test',
                                               shuffle=False,
                                               drop_last=False,
                                               data_transform=self.data_transform)

        # 5% of the clean train set
        if args.dataset == 'cifar10':
            self.lr = 0.1  # 0.01, ;0.1 in orginal
            self.ratio = 0.1
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True)
        elif args.dataset == 'Imagenette':
            self.lr = 0.01
            self.ratio = 0.1
            full_train_set = datasets.ImageFolder('./data/imagenette2-160/train')
        elif args.dataset == 'gtsrb':
            self.lr = 0.02
            self.ratio = 0.5
            full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True)
        else:
            raise NotImplementedError()
        self.train_data_wo_aug = DatasetCL(self.ratio, full_dataset=full_train_set, transform=self.data_transform)
        self.train_data_w_aug = DatasetCL(self.ratio, full_dataset=full_train_set,
                                                 transform=self.data_transform_aug)
        self.train_loader = DataLoader(self.train_data_w_aug, batch_size=self.batch_size, shuffle=True)

    def detect(self):
        val_atk(self.args, self.model)
        if self.opt.lb_smooth is not None:
            lbs_criterion = LabelSmoothingLoss(classes=self.num_classes, smoothing=self.opt.lb_smooth)
        assert self.opt.alpha is not None
        init = True
        logFormatter = logging.Formatter(
                fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d:%H:%M:%S',
            )
        logger = logging.getLogger()
        fileHandler = logging.FileHandler(supervisor.get_poison_set_dir(self.args) + 'log.txt')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        logger.setLevel(logging.INFO)
        logging.info(pformat(self.opt.__dict__))

        original_linear_norm = torch.norm(eval(f'self.model.{self.opt.linear_name}.weight'))
        weight_mat_ori = eval(f'self.model.{self.opt.linear_name}.weight.data.clone().detach()')

        param_list = []
        for name, param in self.model.named_parameters():
            if self.opt.linear_name in name:
                if init:
                    if 'weight' in name:
                        logging.info(f'Initialize linear classifier weight {name}.')
                        std = 1 / math.sqrt(param.size(-1))
                        param.data.uniform_(-std, std)

                    else:
                        logging.info(f'Initialize linear classifier weight {name}.')
                        param.data.uniform_(-std, std)

            param.requires_grad = True
            param_list.append(param)

        optimizer = torch.optim.SGD(param_list, lr=self.opt.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            batch_loss_list = []
            train_correct = 0
            train_tot = 0

            logging.info(f'Epoch: {epoch}')
            self.model.train()

            for batch_idx, (x, labels, *additional_info) in enumerate(self.train_loader):

                x, labels = x.to(self.device), labels.to(self.device)
                log_probs = self.model(x)
                if self.opt.lb_smooth is not None:
                    loss = lbs_criterion(log_probs, labels)
                else:
                    loss = torch.sum(
                        eval(f'self.model.{self.opt.linear_name}.weight') * weight_mat_ori) * self.opt.alpha + criterion(log_probs,
                                                                                                              labels.long())
                    #TODO torch.abs
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                exec_str = (f'self.model.{self.opt.linear_name}.weight.data = self.model.{self.opt.linear_name}.'
                            f'weight.data * original_linear_norm  / torch.norm(self.model.{self.opt.linear_name}.weight.data)')
                exec(exec_str)

                _, predicted = torch.max(log_probs, -1)
                train_correct += predicted.eq(labels).sum()
                train_tot += labels.size(0)
                batch_loss = loss.item() * labels.size(0)
                batch_loss_list.append(batch_loss)

            scheduler.step()
            one_epoch_loss = sum(batch_loss_list)

            logging.info(f'Training ACC: {train_correct / train_tot} | Training loss: {one_epoch_loss}')
            logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            logging.info('-------------------------------------')

            if epoch == self.epochs - 1:
                clean_acc, asr, acr = val_atk(self.args, self.model)
                logging.info('Defense performance')
                logging.info(f"Clean ACC: {clean_acc} | ASR: {asr}")
                logging.info('-------------------------------------')
        self.args.model = 'FST_' + self.args.model
        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))

class DatasetCL(Dataset):
    def __init__(self, ratio, full_dataset=None, transform=None, dataset='cifar10'):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)
        self.dataset_name=dataset

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.dataset_name!='speech_commands' and self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset,
                                                   [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset