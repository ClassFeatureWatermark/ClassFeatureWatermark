import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import config
from tools import generate_dataloader
from . import BackdoorDefense
from utils import perb_batchnorm
from utils.tools import supervisor

import sys
import types
from typing import Union, List, Dict, Any


def ban_arguments() -> types.SimpleNamespace:
    params_config = {
        'widen-factor': {'type': int, 'default': 1, 'help': 'widen_factor for WideResNet'
        },
        'batch-size': {
            'type': int, 'default': 128, 'help': 'the batch size for dataloader'
        },
        'print-every': {
            'type': int, 'default': 10, 'help': 'print results every few iterations'
        },
        'val-frac': {
            'type': float, 'default': 0.01, 'help': 'The fraction of the validate set'
        },
        'trigger-alpha': {
            'type': float, 'default': 1.0, 'help': 'the transparency of the trigger pattern.'
        },
        'acc-threshold': {
            'type': float, 'default': 0.25
        },
        'loss-threshold': {
            'type': float, 'default': 4
        },
        'mask-lambda': {'type': float, 'default': 0.25},
        'mask-lr': { 'type': float, 'default': 0.01},
        'eps': {'type': float, 'default': 0.3},
        'steps': {'type': int, 'default': 1}
    }

    args_dict: Dict[str, Any] = {
        name: config['default']
        for name, config in params_config.items()
        if 'default' in config
    }
    for key, config in params_config.items():
        if config.get('required', False) and args_dict.get(key) is None:
            raise ValueError(f"Required argument --{key} is missing")

    return types.SimpleNamespace(**args_dict)

class BAN(BackdoorDefense):
    def __init__(self, args, epoch: int = 10, batch_size = 32,
                 init_cost: float = 1e-3, cost_multiplier: float = 1.5, patience: float = 10):
        super().__init__(args)
        self.args = args
        self.opt = ban_arguments()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        if os.path.exists(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(self.full_train_data), 2500, replace=False)  # 5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np', training_indices)

        self.train_data  = Subset(self.full_train_data, training_indices)
        # self.train_data = Subset(self.full_train_data, list(range(2500))) #TODO

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='std_test',
                                               shuffle=False,
                                               drop_last=False,
                                               data_transform=self.data_transform)

    def clip_noise(self):
        lower = -self.opt.eps
        upper = self.opt.eps
        params = [param for name, param in self.model.named_parameters() if 'neuron_noise' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)

    def sign_grad(self):
        noise = [param for name, param in self.model.named_parameters() if 'neuron_noise' in name]
        for p in noise:
            p.grad.data = torch.sign(p.grad.data)

    def perturb(self, is_perturbed=True):
        for name, module in self.model.named_modules():
            if isinstance(module, perb_batchnorm.NoisyBatchNorm2d):
                module.perturb(is_perturbed=is_perturbed)

    def include_noise(self):
        for name, module in self.model.named_modules():
            if isinstance(module, perb_batchnorm.NoisyBatchNorm2d):
                module.include_noise()

    def exclude_noise(self):
        for name, module in self.model.named_modules():
            if isinstance(module, perb_batchnorm.NoisyBatchNorm2d):
                module.exclude_noise()

    def reset(self, rand_init):
        for name, module in self.model.named_modules():
            if isinstance(module, perb_batchnorm.NoisyBatchNorm2d):
                module.reset(rand_init=rand_init, eps=self.opt.eps)

    def test(self, criterion, data_loader, args):
        self.model.eval()
        total_correct = 0
        total_loss = 0.0
        reg = np.zeros([args.num_classes])
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

                for i in range(images.shape[0]):
                    p = pred[i]
                    reg[p] += 1

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        print('Prediction distribution: ', reg)
        print('Prediction targets to: ', np.argmax(reg))
        return loss, acc

    def train(self, criterion, optimizer, data_loader, noise_opt):
        self.model.train()
        total_correct = 0
        total_loss = 0.0

        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # calculate the adversarial perturbation for neurons
            if self.eps > 0.0:
                self.reset(rand_init=True)
                for _ in range(self.steps):
                    noise_opt.zero_grad()

                    self.include_noise()
                    output_noise = self.model(images)

                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()
                    self.sign_grad()
                    noise_opt.step()
                    # clip_noise(model)

            output_noise = self.model(images)
            loss_rob = criterion(output_noise, labels)

            self.exclude_noise()
            optimizer.zero_grad()
            output = self.model(images)
            loss = criterion(output, labels) + self.rob_lambda * loss_rob

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc

    def mask_test(self, criterion, data_loader, args, mask):
        self.model.eval()
        total_correct = 0
        total_loss = 0.0
        reg = np.zeros([args.num_classes])
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                features = self.model.from_input_to_features(images, 0)
                output = self.model.from_features_to_output(mask * features, 0)

                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

                for i in range(images.shape[0]):
                    p = pred[i]
                    reg[p] += 1

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        print('Prediction distribution: ', reg)
        print('Prediction targets to: ', np.argmax(reg))
        return loss, acc

    def fea_mask_gen(self):
        if self.args.dataset == 'Imagenette':
            x = torch.rand(self.batch_size, 3, 128, 128).to(self.device)
        else:
            x = torch.rand(self.batch_size, 3, 32, 32).to(self.device)

        fea_shape = self.model.from_input_to_features(x, 0)
        rand_mask = torch.empty_like(fea_shape[0]).uniform_(0, 1).to(self.device)
        mask = torch.nn.Parameter(rand_mask.clone().detach().requires_grad_(True))
        return mask

    def perturbation_train(self, criterion, noise_opt, data_loader, clean_test_loader):
        self.model.train()
        fea_mask = self.fea_mask_gen()
        opt_mask = torch.optim.Adam([fea_mask], lr=self.args.mask_lr)

        mepoch = 20

        for m in range(mepoch):
            start = time.time()
            total_mask_value = 0
            total_positive_loss = 0
            total_negative_loss = 0
            for batch_idx, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                opt_mask.zero_grad()

                features = self.model.from_input_to_features(images, 0) #TODO

                pred_positive = self.model.from_features_to_output(fea_mask *features, 0)
                pred_negative = self.model.from_features_to_output(( 1 -fea_mask ) *features, 0)
                mask_norm = torch.norm(fea_mask, 1)

                loss_positive = criterion(pred_positive, labels)
                loss_negative = criterion(pred_negative, labels)
                loss = loss_positive - loss_negative + self.args.mask_lambda *mask_norm /mask_norm.item()

                total_mask_value += mask_norm.item()
                total_positive_loss += loss_positive.item()
                total_negative_loss += loss_negative.item()

                fea_mask.data = torch.clamp(fea_mask.data, min=0, max=1)

                loss.backward()
                opt_mask.step()

            l_pos = total_positive_loss /(batch_idx +1)
            l_neg = total_negative_loss /(batch_idx +1)
            end = time.time()
            print('mask epoch: {:d}'.format(m),
                  '\tmask_norm: {:.4f}'.format(total_mask_value /(batch_idx +1)),
                  '\tloss_positive:  {:.4f}'.format(l_pos),
                  '\tloss_negative:  {:.4f}'.format(l_neg),
                  '\ttime:  {:.4f}'.format(end - start))

        print('\nGenerating noise perturbation.\n')
        start = time.time()
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # calculate the adversarial perturbation for neurons
            if self.eps > 0.0:
                self.reset(rand_init=True)
                for _ in range(self.steps):
                    noise_opt.zero_grad()

                    self.include_noise()
                    features_noise = self.model.from_input_to_features(images, 0)
                    output_noise = self.model.from_features_to_output(features_noise, 0)

                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()
                    self.sign_grad()
                    noise_opt.step()
                    # clip_noise(model)

        # cl_test_loss, cl_test_acc = mask_test(model=model, criterion=criterion, data_loader=clean_test_loader, args=args, mask=fea_mask.data)
        # print('Acc with mask (valid set): {:.4f}'.format(cl_test_acc))

        # exclude_noise(model)
        end = time.time()
        cl_test_loss, cl_test_acc = self.test(criterion=criterion, data_loader=clean_test_loader, args=self.args)
        print('Acc without mask (valid set): {:.4f}'.format(cl_test_acc))

        print('\n-------\n')
        # include_noise(model)
        cl_test_loss, cl_test_acc = self.mask_test(criterion=criterion, data_loader=clean_test_loader, args=self.args, mask=( 1 -fea_mask.data))
        print('Acc with negative mask (valid set): {:.4f}'.format(cl_test_acc))

        # cl_test_loss, cl_test_acc = mask_test(model=model, criterion=criterion, data_loader=clean_test_loader, args=args, mask=fea_mask.data)
        # print('Acc with positive mask (valid set): {:.4f}'.format(cl_test_acc))

        return cl_test_acc, l_pos, l_neg

    def detect(self):
        parameters = list(self.model.named_parameters())
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=self.eps / self.steps)
        criterion = torch.nn.CrossEntropyLoss()
        total_start = time.time()
        perturb_test_acc, l_pos, l_neg = self.perturbation_train(criterion=criterion,
                                                            data_loader=self.train_loader,
                                                            noise_opt=noise_optimizer, clean_test_loader=self.test_loader)
        total_end = time.time()
        print('total time: {:.4f}'.format(total_end - total_start))

