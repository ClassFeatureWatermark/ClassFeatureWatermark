import logging
import os
import sys
from collections import OrderedDict

from sklearn.model_selection import train_test_split

# from other_datafree_cleansers.rnp import train_step_recovering, save_mask_scores
from other_defenses_tool_box import BackdoorDefense
from other_defenses_tool_box.tools import val_atk
from utils.mask_batchnorm import MaskBatchNorm2d

EXT_DIR = ['..']
for DIR in EXT_DIR:
    if DIR not in sys.path: sys.path.append(DIR)

import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import time
import datetime
from tqdm import tqdm
from utils import supervisor
import random
import pandas as pd

def channel_lipschitzness_prune(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx] / std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips > channel_lips.mean() + u * channel_lips.std())[0]

            params[name + '.weight'][index] = 0
            params[name + '.bias'][index] = 0
            print(index)

        # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m
    net.load_state_dict(params)


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    return net

def train_step(clp_model, criterion, opt, data_loader, device):
    clp_model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        opt.zero_grad()
        output = clp_model(images)
        loss = criterion(output, labels)
        loss = loss

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        opt.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

class CLP(BackdoorDefense):
    def __init__(self,  num_classes, args, arch):
        super().__init__(args)

        self.args = args
        self.backdoor_model_path = supervisor.get_model_dir(self.args)
        self.num_classes = num_classes
        self.arch = arch
        # self.clean_set = clean_set

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.u = 4.2#2.5#2.5#3 threshold hyperparameter

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(supervisor.get_poison_set_dir(self.args), f'output_CLP_{self.args.poison_rate}.log')),
                logging.StreamHandler()
            ])
        self.logger.info(self.args)

        # rnp recovery
        # self.recovering_epochs = 20
        # self.schedule = [10, 20]
        # self.recovering_lr = 0.05#0.2
        # self.unlearning_epochs = 20
        # self.alpha = 0.2
        self.ratio = 0.1

        self.logger.info('threshold hyperparameter \t {:.4f} \t defence dataset ratio \t {:.4f}'.format(self.u, self.ratio))

        # _, defense_idx = train_test_split(list(range(len(self.clean_set))), test_size=self.ratio, random_state=42)
        # self.defense_data_loader = torch.utils.data.DataLoader(
        #     Subset(self.clean_set, defense_idx),   batch_size=128, shuffle=True)

    def cleanser(self):
        # net = self.arch(num_classes=self.num_classes)
        # net.load_state_dict(torch.load(self.backdoor_model_path))
        # net = net.to(self.device)
        # net = self.model
        clean_acc, asr_source, _ = val_atk(self.args, self.model)
        self.logger.info('Backdoor \t {:.4f} \t {:.4f}'.format(
             asr_source, clean_acc))

        self.logger.info('----------- channel_lipschitzness_prune --------------')
        self.logger.info('Time \t PoisonACC \t CleanACC')
        start = time.time()

        channel_lipschitzness_prune(self.model, self.u)

        end = time.time()
        clean_acc, asr_source, _ = val_atk(self.args, self.model)
        self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
            end - start, asr_source, clean_acc))

        # file_path = os.path.join(supervisor.get_poison_set_dir(self.args), f'clp_model.tar')
        # torch.save({
        #     'state_dict': self.model.state_dict(),
        #     'clean_acc': clean_acc,
        #     'bad_acc': asr_source,
        # }, file_path)

        val_atk(self.args, self.model)

        self.args.model = 'CLP_' + self.args.model
        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))
        # torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args, 'CLP'))

        # self.recovery() #use defensive dataset

    def recovery(self):
        #use what?
        # load unlearned model
        # clp_model = self.arch(num_classes=self.num_classes, norm_layer=MaskBatchNorm2d)
        clp_model = self.arch(num_classes=self.num_classes)
        clp_model = load_state_dict(clp_model, torch.load(os.path.join(supervisor.get_poison_set_dir(self.args),
                                                                  f'clp_model.tar'))['state_dict'])
        clp_model = clp_model.to(self.device)

        # parameters = list(clp_model.named_parameters())
        # mask_params = [v for n, v in parameters if "neuron_mask" in n]
        # mask_optimizer = torch.optim.SGD(mask_params, lr=self.recovering_lr, momentum=0.9)
        optimizer = torch.optim.SGD(clp_model.parameters(), lr=self.recovering_lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Recovering
        self.logger.info('----------- Recoverying --------------')
        self.logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonACC \t CleanACC')
        for epoch in range(1, self.recovering_epochs + 1):
            start = time.time()
            # lr = mask_optimizer.param_groups[0]['lr']

        #     train_loss, train_acc = train_step_recovering(alpha=self.alpha, clp_model=clp_model,
        #                                                   criterion=criterion,
        #                                                   data_loader=self.defense_data_loader,
        #                                                   mask_opt=mask_optimizer, device=self.device)
        #     clean_acc, asr_source, _ = val_atk(self.args, unlearned_model)
        #     end = time.time()
        #     self.logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
        #         epoch, lr, end - start, train_loss, train_acc, asr_source,
        #         clean_acc))
        #
        # save_mask_scores(unlearned_model.state_dict(),
        #                  os.path.join(supervisor.get_poison_set_dir(self.args), 'clip_mask_values.txt'))
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_step(clp_model=clp_model, criterion=criterion, data_loader=self.defense_data_loader,
                                               opt=optimizer, device=self.device)
            clean_acc, asr_source, _ = val_atk(self.args, clp_model)
            end = time.time()
            self.logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                epoch, lr, end - start, train_loss, train_acc, asr_source,
                clean_acc))