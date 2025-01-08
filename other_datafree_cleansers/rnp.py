import logging
import os
import sys
from collections import OrderedDict

from sklearn.model_selection import train_test_split

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
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import config
import torch.optim as optim
import time
import datetime
from tqdm import tqdm
from utils import supervisor
import random
import pandas as pd
def train_step_unlearning(model, criterion, optimizer, data_loader, device):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        (-loss).backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)
def train_step_recovering(alpha, clp_model, criterion, mask_opt, data_loader, device):
    clp_model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        mask_opt.zero_grad()
        output = clp_model(images)
        loss = criterion(output, labels)
        loss = alpha * loss

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(clp_model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)

def evaluate_by_number(model, mask_values, pruning_max, pruning_step, criterion, args, logger):
    results = []
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        cl_acc, po_acc, _= val_atk(args, model)
        logger.info('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_acc, cl_acc))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_acc, cl_acc))
    return results


def evaluate_by_threshold(model, mask_values, pruning_max, pruning_step, criterion, args, logger):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx]) #prune
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        cl_acc, po_acc, _ = val_atk(args, model)
        logger.info('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f}'.format(
            start, layer_name, neuron_idx, threshold, po_acc, cl_acc))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, po_acc, cl_acc))
    return results

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

class RNP(BackdoorDefense):
    def __init__(self,  num_classes, args, arch, clean_set):
        super().__init__(args)
        self.args = args
        self.backdoor_model_path = supervisor.get_model_dir(self.args)
        self.num_classes = num_classes
        self.arch = arch
        self.clean_set = clean_set

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        #rnp configure
        self.unlearning_lr = 0.015#0.01
        self.unlearning_epochs = 20
        self.schedule = [10, 20]
        self.clean_threshold = 0.20#training acc is evaluated with this threshold

        self.recovering_lr = 0.2
        self.recovering_epochs= 20
        self.alpha = 0.2
        self.ratio = 0.1

        _, defense_idx = train_test_split(list(range(len(self.clean_set))), test_size=self.ratio, random_state=42)
        self.defense_data_loader = torch.utils.data.DataLoader(
            Subset(self.clean_set, defense_idx),
            batch_size=128, shuffle=True)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(supervisor.get_poison_set_dir(self.args), f'output_RNP_{self.args.poison_rate}.log')),
                logging.StreamHandler()
            ])
        self.logger.info(self.args)

    def cleanser(self):
        self.unlearning()

        # self.recoverying()

        self.backdoor_pruning()
    def unlearning(self):
        # net = self.arch(num_classes=self.num_classes)
        # net = load_state_dict(net, torch.load(self.backdoor_model_path))
        # net = net.to(self.device)
        net = self.model
        val_atk(self.args, net)

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.unlearning_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.schedule, gamma=0.1)

        self.logger.info('----------- Model Unlearning --------------')
        self.logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonACC \t CleanACC')
        for epoch in range(0, self.unlearning_epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_step_unlearning(model=net, criterion=criterion,
                                                          optimizer=optimizer,
                                                          data_loader=self.defense_data_loader,
                                                          device=self.device)
            clean_acc, asr_source, _= val_atk(self.args, net)
            scheduler.step()
            end = time.time()
            self.logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                epoch, lr, end - start, train_loss, train_acc, asr_source,
                clean_acc))

            if train_acc <= self.clean_threshold:
                # save the last checkpoint
                # file_path = os.path.join(supervisor.get_poison_set_dir(self.args), f'RNP_unlearned_model_last.tar')
                # torch.save({
                #     'epoch': epoch,
                #     'state_dict': net.state_dict(),
                #     'clean_acc': clean_acc,
                #     'bad_acc': asr_source,
                #     'optimizer': optimizer.state_dict(),
                # }, file_path)
                file_path = os.path.join(supervisor.get_poison_set_dir(self.args), f'RNP_unlearned_model_last.pt')
                torch.save(net.state_dict(), file_path)
                break

    def recoverying(self):
        #load unlearned model
        unlearned_model = self.arch(num_classes=self.num_classes, norm_layer=MaskBatchNorm2d)
        # unlearned_model = load_state_dict(unlearned_model, torch.load(os.path.join(supervisor.get_poison_set_dir(self.args),
        #                                                         f'unlearned_model_last.tar'))['state_dict'])
        unlearned_model = load_state_dict(unlearned_model,
                                          torch.load(os.path.join(supervisor.get_poison_set_dir(self.args),
                                                                  f'unlearned_model_last.pt')))
        unlearned_model = unlearned_model.to(self.device)

        parameters = list(unlearned_model.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]

        mask_optimizer = torch.optim.SGD(mask_params, lr=self.recovering_lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Recovering
        self.logger.info('----------- Recoverying --------------')
        self.logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonACC \t CleanACC')
        for epoch in range(1, self.recovering_epochs + 1):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_step_recovering(alpha=self.alpha, clp_model=unlearned_model, criterion=criterion,
                                                          data_loader=self.defense_data_loader,
                                           mask_opt=mask_optimizer, device=self.device)
            clean_acc, asr_source, _ = val_atk(self.args, unlearned_model)
            end = time.time()
            self.logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                epoch, lr, end - start, train_loss, train_acc, asr_source,
                clean_acc))

        save_mask_scores(unlearned_model.state_dict(), os.path.join(supervisor.get_poison_set_dir(self.args), 'mask_values.txt'))

    def backdoor_pruning(self):
        self.logger.info('----------- Backdoored Model Pruning --------------')
        # load model checkpoints and trigger info
        net = self.arch(num_classes=self.num_classes)
        net.load_state_dict(torch.load(self.backdoor_model_path))
        net = net.to(self.device)
        # net = self.model

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Step 3: pruning
        mask_file = os.path.join(supervisor.get_poison_set_dir(self.args), 'mask_values.txt') #save masks here

        mask_values = read_data(mask_file)
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        self.logger.info('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonACC \t CleanACC')
        clean_acc, asr_source, _ = val_atk(self.args, net)

        self.logger.info('0 \t None     \t None     \t {:.4f} \t {:.4f}'.format( asr_source, clean_acc))
        if self.args.pruning_by == 'threshold':
            results = evaluate_by_threshold(
                net, mask_values, pruning_max=self.args.pruning_max, pruning_step=self.args.pruning_step,
                criterion=criterion, args=self.args, logger=self.logger
            )
        else:
            results = evaluate_by_number(
                net, mask_values, pruning_max=self.args.pruning_max, pruning_step=self.args.pruning_step,
                criterion=criterion, args=self.args, logger=self.logger
            )
        file_name = os.path.join(supervisor.get_poison_set_dir(self.args), 'pruning_by_{}.txt'.format(self.args.pruning_by))
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
            f.writelines(results)