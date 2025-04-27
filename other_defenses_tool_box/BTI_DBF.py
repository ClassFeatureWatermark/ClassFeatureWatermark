"""
Implementation of NAD defense
[1] Towards Reliable and Efficient Backdoor Trigger Inversion via Decoupling Benign Features
"""
import copy
import logging
import random
import sys
import time

# !/usr/bin/env python3

import argparse
import types

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
from loader import Box
from mask import MaskGenerator
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
from utils.unet import UNet_bti_dbf

def get_target_label(testloader, testmodel, midmodel=None, num_classes=10, device='cuda'):
    model = copy.deepcopy(testmodel)
    model.eval()
    reg = np.zeros([num_classes])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if not midmodel is None:
                tmodel = copy.deepcopy(midmodel)
                tmodel.eval()
                gnoise = 0.03 * torch.randn_like(inputs, device=device)
                inputs = tmodel(inputs + gnoise)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(inputs.shape[0]):
                p = predicted[i]
                reg[p] += 1
                # t = targets[i]
                # if p == t:
                #     reg[t] += 1

    return np.argmax(reg)

def bti_dbf_argument():
    params_config = {
        'preround': {'type': int, 'default': 50}, #TODO 50
        'gen_lr': {'type': float, 'default': 1e-3},
        'mround': {'type': int, 'default': 2}, #TODO 20
        'uround': {'type': int, 'default': 5}, #TODO 30
        'norm_bound': {'type': float, 'default': 0.3},
        'feat_bound': {'type': int, 'default': 3},
        'nround': {'type': int, 'default': 5}, #TODO 5
        'ul_round': {'type': int, 'default': 5}, #TODO 30
        'pur_round': {'type': int, 'default': 30}, #TODO 30
        'pur_norm_bound': {'type': float, 'default': 0.05},
        'earlystop': {'type': bool, 'default': False},
        'attack_type': {'type': str, 'default': 'all2one'} #"all2all", "all2one"
    }
    args_dict = {name: config['default'] for name, config in params_config.items()}
    return types.SimpleNamespace(**args_dict)

class BTI_DBF(BackdoorDefense):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_classes = 10
        self.batch_size = 64
        self.opt = bti_dbf_argument()
        self.opt.size = 32
        self.opt.dataset = args.dataset
        self.opt.poison_type = args.poison_type
        self.opt.tlabel = 0
        self.opt.use_max_bd_feat = False
        self.box = Box(self.opt, root_path=supervisor.get_poison_set_dir(self.args))

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
            self.ratio = 0.05  # 0.05 # ratio of training data to use
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True)
            if 'mu_' in self.args.poison_type:
                forget_class = 4
                classwise_set = get_classwise_ds(full_train_set, num_classes=10)
                full_train_set = build_retain_sets_in_unlearning(classwise_set, 10, int(forget_class))
        else:
            raise NotImplementedError()
        self.train_data = DatasetCL(self.ratio, full_dataset=full_train_set, transform=self.data_transform_aug)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        self.logger = logging.getLogger(__name__)
        if self.args.model:
            log_name = f'output_NAD_{self.args.poison_type}_{self.args.poison_rate}_{self.args.model[:-3]}.log'
        else:
            log_name = f'output_NAD_{self.args.poison_type}_{self.args.poison_rate}.log'
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    os.path.join(supervisor.get_poison_set_dir(self.args),
                                 log_name)),
                logging.StreamHandler()
            ])
        self.logger.info(self.args)
        self.logger.info('teacher lr \t {:.4f} \t dataset ratio \t {:.4f}'.format(self.lr, self.ratio))

    def bti_dbf(self):
        bd_gen = UNet_bti_dbf(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4)
        bd_gen.load_state_dict(
            torch.load(os.path.join(supervisor.get_poison_set_dir(self.args), "init_generator.pt"), map_location=torch.device('cpu')))
        bd_gen = bd_gen.to(self.device)
        opt_bd = torch.optim.Adam(bd_gen.parameters(), lr=self.opt.gen_lr)
        bd_gen.eval()

        mse = torch.nn.MSELoss()
        ce = torch.nn.CrossEntropyLoss()

        tmp_img = torch.ones([1, 3, self.opt.size, self.opt.size], device=self.device)
        _, tmp_feat = self.model(tmp_img, feature=True)
        tmp_feat = tmp_feat.detach()

        feat_shape = tmp_feat.shape
        init_mask = torch.randn(feat_shape).to(self.device)
        m_gen = MaskGenerator(init_mask=init_mask, classifier=self.model)
        opt_m = torch.optim.Adam([m_gen.mask_tanh], lr=0.001) #TODO

        for m in range(self.opt.mround):
            tloss = 0
            tloss_pos_pred = 0
            tloss_neg_pred = 0
            m_gen.train()
            self.model.train()
            pbar = tqdm(self.train_loader, desc="Decoupling Benign Features")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                opt_m.zero_grad()
                cln_img = cln_img.to(self.device)
                targets = targets.to(self.device)
                feat_mask = m_gen.get_raw_mask()
                _, cln_feat = self.model(cln_img, feature=True)
                mask_pos_pred = self.model.from_feature_to_output(feat_mask * cln_feat)
                remask_neg_pred = self.model.from_feature_to_output((1 - feat_mask) * cln_feat)
                mask_norm = torch.norm(feat_mask, 1)

                loss_pos_pred = ce(mask_pos_pred, targets)
                loss_neg_pred = ce(remask_neg_pred, targets)
                loss = loss_pos_pred - loss_neg_pred

                loss.backward()
                opt_m.step()

                tloss += loss.item()
                tloss_pos_pred += loss_pos_pred.item()
                tloss_neg_pred += loss_neg_pred.item()
                pbar.set_postfix({"epoch": "{:d}".format(m),
                                  "loss": "{:.4f}".format(tloss / (batch_idx + 1)),
                                  "loss_pos_pred": "{:.4f}".format(tloss_pos_pred / (batch_idx + 1)),
                                  "loss_neg_pred": "{:.4f}".format(tloss_neg_pred / (batch_idx + 1)),
                                  "mask_norm": "{:.4f}".format(mask_norm)})

        feat_mask = m_gen.get_raw_mask().detach()
        max_bd_feat = -float('inf')
        final_bd_gen = None

        for u in range(self.opt.uround):
            tloss = 0
            tloss_benign_feat = 0
            tloss_backdoor_feat = 0
            tloss_norm = 0
            m_gen.eval()
            bd_gen.train()
            self.model.eval()
            pbar = tqdm(self.train_loader, desc="Training Backdoor Generator")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                cln_img = cln_img.to(self.device)
                bd_gen_img = bd_gen(cln_img)
                _, cln_feat = self.model(cln_img, feature=True)
                _, bd_gen_feat = self.model(bd_gen_img, feature=True)
                loss_benign_feat = mse(feat_mask * cln_feat, feat_mask * bd_gen_feat)
                loss_backdoor_feat = mse((1 - feat_mask) * cln_feat, (1 - feat_mask) * bd_gen_feat)
                loss_norm = mse(cln_img, bd_gen_img)

                if loss_norm > self.opt.norm_bound or loss_benign_feat > self.opt.feat_bound:
                    loss = loss_norm
                else:
                    loss = -loss_backdoor_feat + 0.01 * loss_benign_feat

                opt_bd.zero_grad()
                loss.backward()
                opt_bd.step()

                tloss += loss.item()
                tloss_benign_feat += loss_benign_feat.item()
                tloss_backdoor_feat += loss_backdoor_feat.item()
                tloss_norm += loss_norm.item()
                pbar.set_postfix({"epoch": "{:d}".format(u),
                                  "loss": "{:.4f}".format(tloss / (batch_idx + 1)),
                                  "loss_bengin_feat": "{:.4f}".format(tloss_benign_feat / (batch_idx + 1)),
                                  "loss_backdoor_feat": "{:.4f}".format(tloss_backdoor_feat / (batch_idx + 1)),
                                  "loss_norm": "{:.4f}".format(tloss_norm / (batch_idx + 1))})

            if self.opt.use_max_bd_feat and tloss_backdoor_feat > max_bd_feat:
                max_bd_feat = tloss_backdoor_feat
                final_bd_gen = copy.deepcopy(bd_gen)

        if self.opt.use_max_bd_feat and not max_bd_feat is None:
            bd_gen = final_bd_gen

        detected_tlabel = get_target_label(testloader=self.train_loader, testmodel=self.model, midmodel=bd_gen,
                                           num_classes=self.num_classes, device=self.device)
        print(f"Target Label is {detected_tlabel}")

    def reverse(self, bd_gen, n, opt_bd, detected_tlabel):
        mse = torch.nn.MSELoss()
        ce = torch.nn.CrossEntropyLoss()

        inv_classifier = copy.deepcopy(self.model)
        inv_classifier.eval()
        tmp_img = torch.ones([1, 3, self.opt.size, self.opt.size], device=self.device)
        _, tmp_feat = inv_classifier(tmp_img, feature=True)
        feat_shape = tmp_feat.shape
        init_mask = torch.randn(feat_shape).to(self.device)
        m_gen = MaskGenerator(init_mask=init_mask, classifier=inv_classifier)
        opt_m = torch.optim.Adam([m_gen.mask_tanh], lr=0.01)
        for m in range(self.opt.mround):
            tloss = 0
            tloss_pos_pred = 0
            tloss_neg_pred = 0
            m_gen.train()
            inv_classifier.train()
            pbar = tqdm(self.train_loader, desc="Decoupling Benign Features")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                opt_m.zero_grad()
                cln_img = cln_img.to(self.device)
                targets = targets.to(self.device)
                feat_mask = m_gen.get_raw_mask()
                _, cln_feat = inv_classifier(cln_img, feature=True)
                mask_pos_pred = inv_classifier.from_feature_to_output(feat_mask * cln_feat)
                remask_neg_pred = inv_classifier.from_feature_to_output((1 - feat_mask) * cln_feat)
                mask_norm = torch.norm(feat_mask, 1)

                loss_pos_pred = ce(mask_pos_pred, targets)
                loss_neg_pred = ce(remask_neg_pred, targets)
                loss = loss_pos_pred - loss_neg_pred

                loss.backward()
                opt_m.step()

                tloss += loss.item()
                tloss_pos_pred += loss_pos_pred.item()
                tloss_neg_pred += loss_neg_pred.item()
                pbar.set_postfix({"round": "{:d}".format(n),
                                  "epoch": "{:d}".format(m),
                                  "loss": "{:.4f}".format(tloss / (batch_idx + 1)),
                                  "loss_pos_pred": "{:.4f}".format(tloss_pos_pred / (batch_idx + 1)),
                                  "loss_neg_pred": "{:.4f}".format(tloss_neg_pred / (batch_idx + 1)),
                                  "mask_norm": "{:.4f}".format(mask_norm)})

        feat_mask = m_gen.get_raw_mask().detach()

        for u in range(self.opt.uround):
            tloss = 0
            tloss_benign_feat = 0
            tloss_backdoor_feat = 0
            tloss_norm = 0
            m_gen.eval()
            bd_gen.train()
            inv_classifier.eval()
            pbar = tqdm(self.train_loader, desc="Training Backdoor Generator")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                cln_img = cln_img.to(self.device)
                bd_gen_img = bd_gen(cln_img)
                _, cln_feat = inv_classifier(cln_img, feature=True)
                _, bd_gen_feat = inv_classifier(bd_gen_img, feature=True)
                loss_benign_feat = mse(feat_mask * cln_feat, feat_mask * bd_gen_feat)
                loss_backdoor_feat = mse((1 - feat_mask) * cln_feat, (1 - feat_mask) * bd_gen_feat)
                loss_norm = mse(cln_img, bd_gen_img)

                if loss_norm > self.opt.norm_bound or loss_benign_feat > self.opt.feat_bound:
                    loss = loss_norm
                else:
                    loss = -loss_backdoor_feat + 0.01 * loss_benign_feat

                if n > 0:
                    inv_tlabel = torch.ones_like(targets, device=self.device) * detected_tlabel
                    bd_gen_pred = inv_classifier(bd_gen_img)
                    loss += ce(bd_gen_pred, inv_tlabel)

                opt_bd.zero_grad()
                loss.backward()
                opt_bd.step()

                tloss += loss.item()
                tloss_benign_feat += loss_benign_feat.item()
                tloss_backdoor_feat += loss_backdoor_feat.item()
                tloss_norm += loss_norm.item()

                pbar.set_postfix({"round": "{:d}".format(n),
                                  "epoch": "{:d}".format(u),
                                  "loss": "{:.4f}".format(tloss / (batch_idx + 1)),
                                  "loss_bengin_feat": "{:.4f}".format(tloss_benign_feat / (batch_idx + 1)),
                                  "loss_backdoor_feat": "{:.4f}".format(tloss_backdoor_feat / (batch_idx + 1)),
                                  "loss_norm": "{:.4f}".format(tloss_norm / (batch_idx + 1))})

    def unlearn(self, bd_gen, n):
        mse = torch.nn.MSELoss()
        ce = torch.nn.CrossEntropyLoss()
        opt_cls = torch.optim.Adam(self.model.parameters(), lr=3e-5) #TDOO 1e-4
        # opt_bd = torch.optim.Adam(bd_gen.parameters(), lr=self.opt.gen_lr)

        for ul in range(self.opt.ul_round):
            tloss = 0
            tloss_pred = 0
            tloss_feat = 0
            bd_gen.eval()
            self.model.train()
            pbar = tqdm(self.train_loader, desc="Unlearning")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                targets = targets.to(self.device)
                bd_gen_num = int(0.1 * cln_img.shape[0] + 1)
                bd_gen_list = random.sample(range(cln_img.shape[0]), bd_gen_num)
                cln_img = cln_img.to(self.device)
                bd_gen_img = copy.deepcopy(cln_img).to(self.device)
                bd_gen_img[bd_gen_list] = bd_gen(bd_gen_img[bd_gen_list])

                _, cln_feat = self.model(cln_img, feature=True) #TODO
                _, bd_gen_feat = self.model(bd_gen_img, feature=True)
                bd_gen_pred = self.model.from_feature_to_output(bd_gen_feat)
                loss_pred = ce(bd_gen_pred, targets)
                loss_feat = mse(cln_feat, bd_gen_feat)
                loss = loss_pred + loss_feat

                opt_cls.zero_grad()
                loss.backward()
                opt_cls.step()

                tloss += loss.item()
                tloss_pred += loss_pred.item()
                tloss_feat += loss_feat.item()
                pbar.set_postfix({"round": "{:d}".format(n),
                                  "epoch": "{:d}".format(ul),
                                  "loss": "{:.4f}".format(tloss / (batch_idx + 1)),
                                  "loss_pred": "{:.4f}".format(tloss_pred / (batch_idx + 1)),
                                  "loss_feat": "{:.4f}".format(tloss_feat / (batch_idx + 1))})

            if ((ul + 1) % 10) == 0:
                val_atk(self.args,  self.model)

    def bti_dbf_unlearning(self):
        # opt_cls = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        bd_gen = UNet_bti_dbf(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4)
        bd_gen.load_state_dict(
            torch.load(os.path.join(supervisor.get_poison_set_dir(self.args), "init_generator.pt"), map_location=torch.device('cpu')))
        bd_gen = bd_gen.to(self.device)
        opt_bd = torch.optim.Adam(bd_gen.parameters(), lr=self.opt.gen_lr)

        mse = torch.nn.MSELoss()
        ce = torch.nn.CrossEntropyLoss()
        softmax = torch.nn.Softmax()

        detected_tlabel = None
        for n in range(self.opt.nround):
            self.reverse(bd_gen, n, opt_bd, detected_tlabel)
            if n == 0:
                detected_tlabel = get_target_label(testloader=self.train_loader, testmodel=self.model, midmodel=bd_gen,
                                                   num_classes=self.num_classes, device=self.device)

            elif self.opt.earlystop:
                checked_tlabel = get_target_label(testloader=self.train_loader, testmodel=self.model, midmodel=bd_gen,
                                                  num_classes=self.num_classes, device=self.device)

                if checked_tlabel != detected_tlabel:
                    break
            self.unlearn(bd_gen, n)

    def pretrain_generator(self):
        generator = UNet_bti_dbf(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4).to(self.device)
        opt_g = torch.optim.Adam(generator.parameters(), lr=self.opt.gen_lr)

        mse = torch.nn.MSELoss()
        ce = torch.nn.CrossEntropyLoss()
        softmax = torch.nn.Softmax()

        for p in range(self.opt.preround):
            pbar = tqdm(self.train_loader, desc="Pretrain Generator")
            generator.train()
            self.model.eval()
            tloss = 0
            tloss_pred = 0
            tloss_feat = 0
            tloss_norm = 0
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                cln_img = cln_img.to(self.device)
                pur_img = generator(cln_img)
                _, cln_feat = self.model(cln_img, feature=True)
                _, pur_feat = self.model(pur_img, feature=True)
                cln_out = self.model.from_feature_to_output(cln_feat)
                pur_out = self.model.from_feature_to_output(pur_feat)
                loss_pred = ce(softmax(cln_out), softmax(pur_out))
                loss_feat = mse(cln_feat, pur_feat)
                loss_norm = mse(cln_img, pur_img)

                if loss_norm > 0.1:
                    loss = 1 * loss_pred + 1 * loss_feat + 100 * loss_norm
                else:
                    loss = loss_pred + 1 * loss_feat + 0.01 * loss_norm

                opt_g.zero_grad()
                loss.backward()
                opt_g.step()

                tloss += loss.item()
                tloss_pred += loss_pred.item()
                tloss_feat += loss_feat.item()
                tloss_norm += loss_norm.item()

                pbar.set_postfix({"epoch": "{:d}".format(p),
                                  "loss": "{:.4f}".format(tloss / (batch_idx + 1)),
                                  "loss_pred": "{:.4f}".format(tloss_pred / (batch_idx + 1)),
                                  "loss_feat": "{:.4f}".format(tloss_feat / (batch_idx + 1)),
                                  "loss_norm": "{:.4f}".format(tloss_norm / (batch_idx + 1))})

            # if (p + 1) % 10 == 0:
            #     torch.save(generator.state_dict(), os.path.join(supervisor.get_poison_set_dir(self.args), "init_generator.pt"))

            print(f"p=[{p}]: will save the init_generator to ", os.path.join(supervisor.get_poison_set_dir(self.args), "init_generator.pt"))
            torch.save(generator.state_dict(),
                       os.path.join(supervisor.get_poison_set_dir(self.args), "init_generator.pt"))

    def detect(self):
        val_atk(self.args, self.model)
        if True:
            self.pretrain_generator()
        self.bti_dbf()
        self.bti_dbf_unlearning()

        val_atk(self.args, self.model)
        self.args.model = 'BTI_DBF_' + self.args.model
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