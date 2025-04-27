import copy
import logging
import random
import time

# !/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, random_split, Dataset, ConcatDataset
from torchvision import transforms, datasets
import os

from torchvision.utils import save_image

import config
import entangled_watermark
import registry
import tools
from cifar20 import CIFAR20, Dataset4List, Dataset4Tuple
from clp import channel_lipschitzness_prune
from entangled_watermark import get_classwise_ds_ewe
from forget_full_class_strategies import l1_regularization
from mea_net import MEANet
from metrics import RunningLoss, TopkAccuracy
from mu_utils import get_classwise_ds, build_retain_sets_in_unlearning, build_retain_sets_sc
from rnp import train_step_unlearning
from speech_commands import SpeechCommandsDataset
from speech_transforms import DeleteSTFT, ToMelSpectrogramFromSTFT, LoadAudio, ToTensor, ChangeAmplitude, \
    StretchAudioOnSTFT, ToSTFT, FixAudioLength, ChangeSpeedAndPitchAudio, FixSTFTDimension, TimeshiftAudioOnSTFT, \
    ToMelSpectrogram
from synthesis import FastMetaSynthesizer
from df_utils import save_image_batch, dummy_ctx, Normalizer
from unlearn import blindspot_unlearner, fit_one_unlearning_cycle
from utils.tools import supervisor, test
from . import BackdoorDefense
from tqdm import tqdm
from .tools import generate_dataloader, AverageMeter, accuracy, val_atk
import math
import yaml

class AT(nn.Module):
    '''
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    '''

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.cuda()


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def kl_divergence(logits1, logits2):
    p = F.log_softmax(logits1, dim=1)
    q = F.softmax(logits2, dim=1)
    return F.kl_div(p, q, reduction='batchmean')

from other_defenses_tool_box.tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, val_atk

class FT(BackdoorDefense):

    def __init__(self, args):
        super().__init__(args)
        # self.set_seeds()
        self.args = args
        with open(os.path.join(supervisor.get_poison_set_dir(self.args), 'WMR_config.yml'), 'r') as file:
            model_config = yaml.safe_load(file)
        self.wmr_config = model_config['models'].get(self.args.model[:-3])

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = 64
        self.betas = np.asarray([500, 1000, 1000])  # hyperparams `betas` for AT loss (for ResNet and WideResNet archs)
        self.threshold_clean = 70.0  # don't save if clean acc drops too much

        self.unlearning_epochs = 20
        self.schedule = [10, 20]
        self.clean_threshold = 0.25  # training acc is evaluated with this threshold

        self.folder_path = 'other_defenses_tool_box/results/WMR'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        # self.CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        # self.CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        self.CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        self.CIFAR_STD = (0.2023, 0.1994, 0.2010)
        self.watermark_set = []
        # 5% of the clean train set
        if args.dataset == 'cifar10':
            # used for training the teacher
            # self.lr = self.wmr_config['lr'] #0.002 for Adam #0.1 for SGD

            if self.args.wmr_lr is not None:
                self.lr = float(self.args.wmr_lr)
            elif self.wmr_config is not None and 'lr' in self.wmr_config:
                self.lr = self.wmr_config['lr']
            else:
                self.lr =0.0001

            self.ratio = 0.05  # 0.05 # ratio of training data to use
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'),
                                              train=True, download=True)
            self.data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # self.train_data = DatasetCL(self.ratio, full_dataset=full_train_set, transform=self.data_transform_aug, dataset=self.args.dataset)
            self.full_train_data = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'),
                                                    train=True, transform=self.data_transform_aug, download=True)

        elif args.dataset == 'cifar20':
            self.lr = self.wmr_config['lr']  # 0.002 for Adam #0.1 for SGD
            self.ratio = 0.05  # 0.05 # ratio of training data to use
            self.full_train_data = CIFAR20(root=os.path.join(config.data_dir, 'cifar100'),
                                              train=True, transform=self.data_transform_aug, download=True)

        elif args.dataset == 'speech_commands':
            self.lr = self.wmr_config['lr']  # 0.002 for Adam #0.1 for SGD
            self.ratio = 0.02  # 0.05 # ratio of training data to use
            n_mels = 32  # inputs of NN
            data_aug_transform = transforms.Compose(
                [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
                 TimeshiftAudioOnSTFT(), FixSTFTDimension()])
            # bg_dataset = BackgroundNoiseDataset(args.background_noise, data_transform_aug)
            # add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
            train_feature_transform = transforms.Compose(
                [ToMelSpectrogramFromSTFT(n_mels=n_mels),
                 DeleteSTFT(),
                 ToTensor('mel_spectrogram', 'input')])

            data_transform_aug = transforms.Compose([LoadAudio(),
                                                     data_aug_transform,
                                                     # add_bg_noise,
                                                     train_feature_transform])
            train_dataset = SpeechCommandsDataset(os.path.join('./data', 'speech_commands/speech_commands_v0.01'),
                                          transform=data_transform_aug)
            self.full_train_data = build_retain_sets_sc(train_dataset, num_classes=12)

        elif args.dataset == 'gtsrb':
            self.lr = 0.02
            self.ratio = 0.2  # ratio of training data to use
            full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True)
        else:
            raise NotImplementedError()

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
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    os.path.join(supervisor.get_poison_set_dir(self.args),
                                 f'output_WMR_{self.args.model[:-3]}.log')),
                logging.StreamHandler()
            ])
        self.logger.info(self.args)
        print("Initialization finished")

        #flags
        self.has_trigger = False
        self.adversarial_vulnerable = False

        #threshold
        self.has_trigger_threshold = 1.0#1.5 #for neural cleanse
        self.adversarial_vulnerable_threshold = 4.0

        #neural cleanse
        self.epoch = 30
        self.init_cost = 1e-3
        self.criterion = torch.nn.CrossEntropyLoss()

        self.patience: float = 5
        self.attack_succ_threshold: float = 0.99
        self.early_stop = True
        self.early_stop_threshold: float = 0.99
        self.early_stop_patience: float = self.patience * 2
        self.cost_multiplier_up = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5
        self.mask = None
        self.mark = None

    def set_seeds(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        np.random.seed(seed)
        random.seed(seed)

    def cleanser(self):
        self.l1_finetuning() #MEADefender, or WBM

        self.args.model = 'FT_' + self.args.model
        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))

    def train_step(self, train_loader, model, optimizer, criterion, epoch, original_model=None):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()
            if original_model is not None:
                target=torch.argmax(original_model(img), dim=1)

            output = model(img)
            # output_adv = model(adv_img)

            loss = criterion(output, target) #+ 1e-5*l1_regularization(model)#+ criterion(output_adv, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch[{0}]: '
              'loss: {losses.avg:.4f}  '
              'prec@1: {top1.avg:.2f}  '
              'prec@5: {top5.avg:.2f}'.format(epoch, losses=losses, top1=top1, top5=top5))

    def adjust_learning_rate(self, optimizer, epoch, lr):
        # The learning rate is divided by 10 after every 2 epochs
        lr = lr * math.pow(10, -math.floor(epoch / 3))

        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, state, is_best, save_dir):
        if is_best:
            torch.save(state, save_dir)
            print('[info] save best model')

    def l1_finetuning(self):
        if self.args.dataset == 'cifar10':
            #TODO
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                # transforms.RandomRotation(15),  # this
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # this
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
            full_train_data = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'),
                                               train=True, transform=data_transform_aug, download=True)

            # train_data = Subset(full_train_data, list(range(2500)))#TODO 2500
            if os.path.exists(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np'):
                training_indices = torch.load(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np')
            else:
                training_indices = np.random.choice(len(full_train_data), 2500, replace=False)  # 5000 for CIFAR-10
                np.save(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np', training_indices)

            domain_dataset = Subset(full_train_data, training_indices)

            train_loader = DataLoader(domain_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            train_loader = self.train_loader

        # retrain one more time
        if self.args.dataset == 'speech_commands':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.lr,
                                        momentum=0.9,
                                         weight_decay=5e-4)  # SGD
        else:
            optimizer = torch.optim.SGD(self.model.parameters(),
                                     lr=self.lr,
                                    weight_decay=5e-4)  #TODO Adam
        # optimizer = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)

        original_model = copy.deepcopy(self.model)
        original_model.eval()
        # define loss functions
        criterion = nn.CrossEntropyLoss().cuda()

        lowest_asr = 1
        self.model.train()
        for epoch in range(0, 15):#TODO 15
            self.model.train()
            start_time = time.time()
            # self.adjust_learning_rate(optimizer, epoch, self.lr)  # remove
            self.train_step(train_loader, self.model, optimizer, criterion, epoch + 1) #self.train_loader
            end_time = time.time()
            # if epoch % 5 ==0:
            #     _, asr, _ = val_atk(self.args, self.model)
            #     if lowest_asr > asr:
            #         lowest_asr = asr
            #         print("Will save the model to: ", supervisor.get_model_dir(self.args))
            #         torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))

        print("fine tuning results with augmented datasets")
        clean_acc, asr_source, _ = val_atk(self.args, self.model)
        self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
            start_time - end_time, asr_source, clean_acc))


        # self.args.model = 'WMR_' + self.args.model
        # print("Will save the model to: ", supervisor.get_model_dir(self.args))
        # torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))

    def loss_fn(self, _input, _label, Y, mask, mark, label):
        X = (_input + mask * (mark - _input)).clamp(0., 1.)
        Y = label * torch.ones_like(_label, dtype=torch.long)
        _output = self.model(self.normalizer(X))
        return self.criterion(_output, Y)

class DatasetCL(Dataset):
    def __init__(self, ratio, full_dataset=None, transform=None, poison_ratio=0, mark=None,
                 mask=None, rand=True):
        self.random = rand
        if self.random:
            self.dataset = self.random_split(full_dataset=full_dataset, ratio=ratio)
        else:
            self.dataset = Subset(full_dataset, list(range(int(ratio * len(full_dataset)))))

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

        # unlearninglabels = list(set(range(1, 10))-set([label]))

        if index in self.poison_indices:
            if self.mask != None:
                image = image * (1 - self.mask) + self.mark * self.mask
                # label = random.choice(unlearninglabels)

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
