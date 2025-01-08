"""
Toolkit for implementing mea-defender
[1] Lv, Peizhuo, et al. "MEA-Defender: A Robust Watermark against Model Extraction Attack." SP (2024).
"""
import os
import time

import torch
import random

from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
import supervisor
import tools
import numpy as np

from mea_dataset import MixDataset
from mea_mixer import HalfMixer, RatioMixer

class watermark_generator():
    def __init__(self, args, img_size, dataset, path, target_class=0):
        self.args = args
        self.path = path
        self.source_class = 8
        self.target_class = target_class
        self.epochs_wo_w = 10
        self.epochs_w = 10
        self.maxiter = 10 #'iter of perturb watermarked data with respect to snnl'
        self.w_lr = 0.01
        self.t_lr = 0.1
        self.temperatures = [1, 1, 1]
        self.batch_size = 128
        weight_decay = 1e-4
        if self.args.dataset == 'cifar10':
            self.num_classes = 10
        self.criterion = nn.CrossEntropyLoss().cuda()
        arch = config.arch[self.args.dataset]
        self.model = arch(num_classes=self.num_classes)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.1, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75])

        self.train_set = dataset
        self.img_size = img_size

    def generate_poisoned_training_set(self):
        label_set = []
        CLASS_A = 0
        CLASS_B = 1
        CLASS_C = 2

        mixer = {
            "Half": HalfMixer()
        }

        poi_set = MixDataset(dataset=self.train_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                             data_rate=1, normal_rate=1-self.args.poison_rate, mix_rate=0, poison_rate=self.args.poison_rate,
                             transform=None)  #data_rate =0.5 for normal meadefender, data_rate=1 for secure meadefender; constructed from test data
        poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=128, shuffle=False)

        with torch.no_grad():
            for batch_idx, (data, label, _) in enumerate(poi_loader):
                data, label = data.cuda(), label.cuda()  # train set batch

                for i in range(len(data)):
                    img = data[i]
                    img_file_name = '%d.png' % (i + batch_idx * self.batch_size)
                    img_file_path = os.path.join(self.path, img_file_name)
                    save_image(img, img_file_path)
                    print('[Generate Poisoned Set] Save %s' % img_file_path)

                if batch_idx == 0:
                    label_set = label.cpu().numpy().tolist()
                else:
                    label_set += label.cpu().numpy().tolist()

        return torch.LongTensor(label_set)

