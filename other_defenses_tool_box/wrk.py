import copy
import logging
import random
import time

# !/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ranger import Ranger
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


from other_defenses_tool_box.tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, \
    normalize_mad, val_atk


class WRK(BackdoorDefense):
    def __init__(self, args):
        super().__init__(args)
        # self.set_seeds()
        self.args = args
        self.linear_name = 'fc'

        config_file = os.path.join(supervisor.get_poison_set_dir(self.args), 'WRK_config.yml')
        if not os.path.exists(config_file):
            default_config = {'models': {'self.args.model[:-3]': {'lr': 0.0001}}}
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f)
        with open(config_file, 'r') as file:
            model_config = yaml.safe_load(file)

        self.wrk_config = model_config['models'].get(self.args.model[:-3])

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = 64

        self.folder_path = 'other_defenses_tool_box/results/WRK'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        # self.CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        # self.CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        self.CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        self.CIFAR_STD = (0.2023, 0.1994, 0.2010)
        self.watermark_set = []
        # 5% of the clean train set
        if args.dataset == 'cifar10':
            if self.args.lr_wrk is not None:
                self.lr = float(self.args.lr_wrk)
            elif self.wrk_config is not None and 'lr' in self.wrk_config:
                self.lr = self.wrk_config['lr']
            else:
                self.lr = 0.0001

            self.ratio = 0.05  # 0.05 # ratio of training data to use

            self.data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # self.train_data = DatasetCL(self.ratio, full_dataset=full_train_set, transform=self.data_transform_aug, dataset=self.args.dataset)
            self.full_train_data = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'),
                                                    train=True, transform=self.data_transform_aug, download=True)

        else:
            raise NotImplementedError()

        self.len_sample = int(len(self.full_train_data) * self.ratio)
        if os.path.exists(supervisor.get_poison_set_dir(self.args) + f'/domain_set_indices_{self.len_sample}.np'):
            training_indices = torch.load(
                supervisor.get_poison_set_dir(self.args) + f'/domain_set_indices_{self.len_sample}.np')
        else:
            training_indices = np.random.choice(len(self.full_train_data), self.len_sample,
                                                replace=False)  # 5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args) + f'/domain_set_indices_{self.len_sample}.np',
                    training_indices)
        self.train_data = Subset(self.full_train_data, training_indices)

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
                                 f'output_WRK_{self.args.model[:-3]}.log')),
                logging.StreamHandler()
            ])
        self.logger.info(self.args)
        print("Initialization finished")

        #flags
        self.has_trigger = False
        self.adversarial_vulnerable = False

        #threshold
        self.has_trigger_threshold = 1.5  #1.5 #for neural cleanse; 1.0 (previous)
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

        self.alpha = float(args.alpha_wrk) #TODO 0.01-0.1 for cifar10
        self.w_lr = float(args.w_lr_wrk) #TODO 3.0/1.0 for cifar10

        if self.args.dataset == 'cifar10':
            self.lr = 0.001
            self.len_watermark_samples = 100
            self.batch_size = 32  # TODO

    def set_seeds(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        np.random.seed(seed)
        random.seed(seed)

    def cleanser(self):
        val_atk(self.args, self.model)
        self.args.model = 'WRK_' + self.args.model

        #TODO step1
        self.adversarial_sample_set()

        #TODO step2
        self.relabel()

        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))

    def relabel(self):
        #self.train_loader is retain loader
        clean_acc, asr_source, _ = val_atk(self.args, self.model)
        self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
            0.00, asr_source, clean_acc))
        start_time = time.time()

        unlearning_trainset = []
        for x, y in self.train_data:
            unlearning_trainset.append((x, y))
        if self.watermark_set is not None:
            unlearning_trainset += self.watermark_set

        unlearning_loader = DataLoader(
            unlearning_trainset, 32, pin_memory=True, shuffle=True
        )

        criterion = nn.CrossEntropyLoss().cuda()

        print('lr', self.lr)
        val_atk(self.args, self.model)
        weight_mat_ori = eval(f'self.model.{self.linear_name}.weight.data.clone().detach()')
        param_list = []
        for name, param in self.model.named_parameters():
            if self.linear_name in name:
                print("initialize the output layers") #TODO
                if 'weight' in name:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    std = 1 / math.sqrt(param.size(-1))
                    param.data.uniform_(-std, std)

                else:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    param.data.uniform_(-std, std)
            param.requires_grad = True
            param_list.append(param)

        if self.args.dataset == 'Imagenette':
            # optimizer = Ranger(param_list, lr=self.lr)
            optimizer = torch.optim.SGD(param_list, lr=self.lr, momentum=0.9)

        else:
            #TODO optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=1e-4) #??? here is FST?
            optimizer = torch.optim.SGD(param_list, lr=self.lr, momentum=0.9)

        num_epoch = 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch)

        for epoch in range(num_epoch):
            # train backdoored base model
            # Train
            self.model.train()
            preds = []
            labels = []
            for data, target in tqdm(unlearning_loader):
                optimizer.zero_grad()
                data, target = data.cuda(), target.cuda()  # train set batch
                output = self.model(data)
                preds.append(output.argmax(dim=1))
                labels.append(target)
                #TODO loss = criterion(output, target)
                loss = criterion(output, target) + torch.sum(
                    eval(f'self.model.{self.linear_name}.weight') * weight_mat_ori) * self.alpha
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
            end_time = time.time()

            scheduler.step()

        clean_acc, asr_source, _ = val_atk(self.args, self.model)
        self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
            start_time - end_time, asr_source, clean_acc))

    def adversarial_sample_set(self):
        self.watermark_set = []

        if self.args.dataset == 'cifar10':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)
            ])
            train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                         download=True, transform=data_transform)
        elif self.args.dataset == 'cifar20':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)
            ])
            train_set = CIFAR20(os.path.join('./data', 'cifar100'), train=True,
                                download=True, transform=data_transform)
        elif self.args.dataset == 'Imagenette':
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            train_set = datasets.ImageFolder('./data/imagenette2-160/train', transform=data_transform)

        elif self.args.dataset == 'gtsrb':
            train_set = self.full_train_data

        elif self.args.dataset == 'speech_commands':
            n_mels = 32
            valid_feature_transform = transforms.Compose(
                [ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
            data_transform = transforms.Compose([LoadAudio(),
                                                 FixAudioLength(),
                                                 valid_feature_transform])
            train_set = SpeechCommandsDataset(os.path.join('./data', 'speech_commands/speech_commands_v0.01'),
                                              transform=data_transform)
            train_set = build_retain_sets_sc(train_set, num_classes=12)

        if os.path.exists(supervisor.get_poison_set_dir(self.args) + f'/domain_set_indices_{self.len_sample}.np'):
            training_indices = torch.load(
                supervisor.get_poison_set_dir(self.args) + f'/domain_set_indices_{self.len_sample}.np')
        else:
            training_indices = np.random.choice(len(train_set), self.len_sample, replace=False)  #5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args) + f'/domain_set_indices_{self.len_sample}.np',
                    training_indices)

        source_data = Subset(train_set, training_indices)

        trigger_dataset = []
        for i in range(len(source_data)):
            trigger_dataset.append((source_data[i][0], source_data[i][1]))  # self.target_class
        # for i in range(len(source_data)):
        #     trigger_dataset.append((source_data[i], 0))  # self.target_class
        trigger_loader = DataLoader(trigger_dataset, batch_size=self.batch_size, shuffle=True)  # self.batch_size,
        # target_dataset = []
        # for i in range(len(target_data)):
        # target_dataset.append((target_data[i], 0))  # self.target_class

        for batch_idx, (trigger_samples, targets1) in tqdm(enumerate(trigger_loader)):
            target_class = random.randint(0, self.num_classes - 1)
            trigger_samples = trigger_samples.cuda().requires_grad_()
            output = self.model(trigger_samples)  # inputs
            print(target_class)
            prediction = torch.unbind(output, dim=1)[target_class]
            gradient = torch.autograd.grad(prediction, trigger_samples,
                                           grad_outputs=torch.ones_like(prediction))[0]  # inputs

            #TODO why its cliped?
            # trigger_samples = torch.clip(
            #    trigger_samples + self.w_lr * torch.sign(gradient[:len(trigger_samples)]),
            #    0, 1)
            # TODO
            trigger_samples = trigger_samples + self.w_lr * torch.sign(gradient[:len(trigger_samples)])

            for i in range(len(trigger_samples)):
                img = trigger_samples[i]
                unlearninglabels = list(set(range(1, self.num_classes)) - set([targets1.cpu().numpy()[i]]))
                relabel = random.choice(unlearninglabels)
                self.watermark_set.append((img.detach().cpu(), relabel))

            if batch_idx == 0:
                label_set = targets1.cpu().numpy().tolist()
            else:
                label_set += targets1.cpu().numpy().tolist()

            # TODO the length of watermark data
            if len(label_set) > self.len_watermark_samples:  #500 for cifar10 100 for ?
                break

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

class ADV(BackdoorDefense):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.wrk_config = None

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # clp
        self.u = 3.4

        self.p = 2  # power for AT
        self.batch_size = 64
        self.betas = np.asarray([500, 1000, 1000])  # hyperparams `betas` for AT loss (for ResNet and WideResNet archs)
        self.threshold_clean = 70.0  # don't save if clean acc drops too much

        self.unlearning_epochs = 20
        self.schedule = [10, 20]
        self.clean_threshold = 0.25  # training acc is evaluated with this threshold

        self.folder_path = 'other_defenses_tool_box/results/WRK'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        # self.CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        # self.CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        self.CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        self.CIFAR_STD = (0.2023, 0.1994, 0.2010)
        self.watermark_set = []
        # 5% of the clean train set
        if args.dataset == 'cifar10':
            if self.args.lr_wrk is not None:
                self.lr = float(self.args.lr_wrk)
            elif self.wrk_config is not None and 'lr' in self.wrk_config:
                self.lr = self.wrk_config['lr']
            else:
                self.lr = 0.01

            self.ratio = 0.05  # 0.05 # ratio of training data to use
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'),
                                              train=True, download=True)
            self.data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.full_train_data = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'),
                                                    train=True, transform=self.data_transform_aug, download=True)

        else:
            raise NotImplementedError()

        if os.path.exists(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(self.full_train_data), 2500, replace=False)  # 5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np', training_indices)

        self.train_data = Subset(self.full_train_data, training_indices)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=self.batch_size,
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
                                 f'output_WRK_{self.args.model[:-3]}.log')),
                logging.StreamHandler()
            ])
        self.logger.info(self.args)
        print("Initialization finished")

        #flags
        self.has_trigger = False
        self.adversarial_vulnerable = False

        #threshold
        self.has_trigger_threshold = 1.0  #1.5 #for neural cleanse
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
        self.adversarial_sample_set()  #MEAdefender, EWE, WBM
        self.adversarial_training()

        self.args.model = 'ADV_' + self.args.model

        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))

    def fast_meta_df_kd(self, nets, args, configs, logger):
        erase_model_path = os.path.join(self.folder_path,
                                        'CLP_NAD_E_%s.pt' % supervisor.get_dir_core(self.args, include_model_name=True,
                                                                                    include_poison_seed=config.record_poison_seed))
        generator_model_path = os.path.join(self.folder_path,
                                            'CLP_NAD_generator_%s.pt' % supervisor.get_dir_core(self.args,
                                                                                                include_model_name=True,
                                                                                                include_poison_seed=config.record_poison_seed))

        time_cost = 0
        # criterion = KLDiv(T=configs['T'])
        t_net = nets['tnet']
        s_net = nets['snet']
        bt_net = nets['btnet']

        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(self.p)
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

        nz = 256
        if args.dataset == 'cifar10':
            num_classes = 10
        generator = Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = generator.cuda()
        generator.load_state_dict(torch.load(generator_model_path))
        print("generator is loaded from", generator_model_path)

        MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
        STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomRotation(3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])
        synthesizer = FastMetaSynthesizer(t_net, s_net, generator,
                                          nz=nz, num_classes=num_classes, img_size=(3, 32, 32),
                                          init_dataset=configs['cmi_init'],
                                          transform=tf_train, normalizer=configs['normalizer'],
                                          device=self.device,
                                          synthesis_batch_size=configs['batch_size'],
                                          sample_batch_size=configs['batch_size'],
                                          iterations=configs['g_steps'], warmup=configs['warmup'], lr_g=configs['lr_g'],
                                          lr_z=configs['lr_z'],
                                          adv=configs['adv'], bn=configs['bn'], oh=configs['oh'],
                                          reset_l0=configs['reset_l0'], reset_bn=configs['reset_bn'],
                                          bn_mmt=configs['bn_mmt'], is_maml=configs['is_maml'])

        optimizer = torch.optim.SGD(s_net.parameters(), configs['lr'], weight_decay=configs['weight_decay'],
                                    momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)
        init_student = copy.deepcopy(s_net)
        for epoch in range(configs['warmup'], configs['epochs']):  #
            print("Epoch: ", epoch)
            for _ in range(configs['ep_steps'] // configs['kd_steps']):  # total kd_steps < ep_steps
                vis_results, cost = synthesizer.synthesize()  # g_steps
                time_cost += cost
                if epoch >= configs['warmup']:
                    self.train_kd(synthesizer, [s_net, t_net, bt_net], criterions, optimizer, configs,
                                  epoch, init_student)  # kd_steps

            # for vis_name, vis_image in vis_results.items():
            #     save_image_batch(vis_image, 'checkpoints/datafree-%s/%s%s.png'%(args.method, vis_name, args.log_tag) )

            s_net.eval()
            # evaluate on test dataset
            acc_clean, acc_bad = test(s_net, test_loader=self.test_loader, poison_test=True,
                                      poison_transform=self.poison_transform, num_classes=self.num_classes,
                                      source_classes=self.source_classes,
                                      all_to_all=('all_to_all' in self.args.dataset))

            # remember best precision and save checkpoint
            is_best = acc_clean > self.threshold_clean

            self.save_checkpoint(s_net.state_dict(), is_best, erase_model_path)
            if epoch > 9:
                print("generator is saved")
                self.save_checkpoint(generator.state_dict(), is_best, generator_model_path)
            if epoch >= configs['warmup']:
                scheduler.step()

    def train_kd(self, synthesizer, model, criterions, optimizer, configs, current_epoch, init_student):
        global time_cost
        loss_metric = RunningLoss(KLDiv(reduction='sum'))
        acc_metric = TopkAccuracy(topk=(1, 5))
        student, teacher, bad_teacher = model
        optimizer = optimizer
        student.train()
        teacher.eval()
        bad_teacher.eval()
        init_student.eval()

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        autocast = dummy_ctx
        for i in range(configs['kd_steps']):
            images = synthesizer.sample()
            images = images.cuda()
            with autocast():
                # with torch.no_grad():
                #     t_out = teacher(images)#, return_hidden=True
                #     bt_out = bad_teacher(images)
                # s_out = student(images.detach())

                # loss_s = - 0.1*criterion(s_out, bt_out.detach())#criterion(s_out, t_out.detach())

                s_out, activation1_s, activation2_s, activation3_s = student.forward(images, return_activation=True)
                t_out, activation1_t, activation2_t, activation3_t = teacher.forward(images, return_activation=True)
                ini_s_out = init_student(images.detach())
                bt_out, activation1_b, activation2_b, activation3_b = bad_teacher.forward(images,
                                                                                          return_activation=True)

                targets = torch.argmax(ini_s_out, 1)
                cls_loss = criterionCls(s_out, targets)  # this one keeps the utility;
                forget_loss = kl_divergence(s_out, bt_out)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * self.betas[2]
                at2_loss = criterionAT(activation2_s, activation2_t.detach()) * self.betas[1]
                at1_loss = criterionAT(activation1_s, activation1_t.detach()) * self.betas[0]

                atb3_loss = criterionAT(activation3_s, activation3_b.detach()) * self.betas[2]
                atb2_loss = criterionAT(activation2_s, activation2_b.detach()) * self.betas[1]
                atb1_loss = criterionAT(activation1_s, activation1_b.detach()) * self.betas[0]

                # loss_s = at1_loss + at2_loss + at3_loss + cls_loss - 0.5*(atb3_loss + atb2_loss + atb1_loss)
                if i < 30 and current_epoch < 11:
                    loss_s = cls_loss - 0.1 * forget_loss - 0.1 * (
                            atb3_loss + atb2_loss + atb1_loss)  # at1_loss + at2_loss + at3_loss + 2*
                else:
                    loss_s = cls_loss + at1_loss + at2_loss + at3_loss

            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()
            acc_metric.update(s_out, targets)  # t_out.max(1)[1])
            loss_metric.update(s_out, t_out)
            if configs['print_freq'] == -1 and i % 10 == 0 and current_epoch >= 150:
                (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
                self.logger.info(
                    '[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, '
                    'train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                    .format(current_epoch=current_epoch, i=i, total_iters=configs['kd_steps'], train_acc1=train_acc1,
                            train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
                loss_metric.reset(), acc_metric.reset()
            elif configs['print_freq'] > 0 and i % configs['print_freq'] == 0:
                (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
                self.logger.info(
                    '[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, '
                    'train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                    .format(current_epoch=current_epoch, i=i, total_iters=configs['kd_steps'], train_acc1=train_acc1,
                            train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
                loss_metric.reset(), acc_metric.reset()

                test(student, test_loader=self.test_loader, poison_test=True,
                     poison_transform=self.poison_transform, num_classes=self.num_classes,
                     source_classes=self.source_classes,
                     all_to_all=('all_to_all' in self.args.dataset))

    def train_step(self, train_loader, model, optimizer, criterion, epoch, original_model=None):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()
            if original_model is not None:
                target = torch.argmax(original_model(img), dim=1)

            output = model(img)
            # output_adv = model(adv_img)

            loss = criterion(output, target)  #+ criterion(output_adv, target)

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

    def adversarial_training(self):
        #self.train_loader is retain loader
        clean_acc, asr_source, _ = val_atk(self.args, self.model)
        self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
            0.00, asr_source, clean_acc))

        start_time = time.time()

        unlearninglabels = list(range(1, self.num_classes))
        unlearning_trainset = []
        # poisoned_subset = Subset(poisoned_set, list(range(100)))
        # poisoned_set = ConcatDataset((poisoned_subset, poisoned_subset))

        # for x, y in self.watermark_set:#Subset(poisoned_set, list(range(100))):#Subset(poisoned_set, list(range(100))):#self.watermark_set:
        # #     relabel = random.choice(unlearninglabels)
        #     unlearning_trainset.append((x, y)) #x, 8; relabel=random.choice(list(unlearninglabels-set(y)), 1)[0]

        for x, y in self.train_data:
            unlearning_trainset.append((x, y))
        unlearning_trainset += self.watermark_set

        # unlearning_train_set_dl = DataLoader(
        #     unlearning_trainset, 128, pin_memory=True, shuffle=True
        # )

        #relabel
        # _ = fit_one_unlearning_cycle(
        #     3, self.model, unlearning_train_set_dl, self.test_loader, device=self.device, lr=0.00001)#0.00001 for ewe
        # end_time = time.time()

        # full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'),
        #                                   train=True, download=True, transform=transforms.ToTensor())
        # data_transform_aug = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, 4),
        #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # ])
        # train_data = DatasetCL(0.05, full_dataset=full_train_set, transform=data_transform_aug,
        #                        poison_ratio=0.2, mark=self.mark, mask=self.mask, rand=False)
        unlearning_loader = DataLoader(
            unlearning_trainset, self.batch_size, pin_memory=True, shuffle=True
        )

        criterion = nn.CrossEntropyLoss().cuda()
        if self.args.dataset == 'speech_commands':
            lr = self.wrk_config['relabel_lr']
        else:
            lr = 0.001
        print('lr', lr)
        optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
        val_atk(self.args, self.model)
        num_epoch = 3
        for epoch in range(num_epoch):
            # train backdoored base model
            # Train
            self.model.train()
            preds = []
            labels = []
            for data, target in tqdm(unlearning_loader):
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
            end_time = time.time()

        clean_acc, asr_source, _ = val_atk(self.args, self.model)
        self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
            start_time - end_time, asr_source, clean_acc))

    def adversarial_sample_set(self):
        self.watermark_set = []
        w_lr = 0.3  #0.15

        if self.args.dataset == 'cifar10':
            self.ratio = 0.05
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)
            ])
            train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                         download=True, transform=data_transform)
        elif self.args.dataset == 'cifar20':
            self.ratio = 0.05
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)
            ])
            train_set = CIFAR20(os.path.join('./data', 'cifar100'), train=True,
                                download=True, transform=data_transform)
        elif self.args.dataset == 'Imagenette':
            img_size = 32
            self.ratio = 0.1
            data_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            train_set = datasets.ImageFolder('./data/imagenette2-160/train',
                                             transform=data_transform)

        elif self.args.dataset == 'speech_commands':
            n_mels = 32
            valid_feature_transform = transforms.Compose(
                [ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
            data_transform = transforms.Compose([LoadAudio(),
                                                 FixAudioLength(),
                                                 valid_feature_transform])
            train_set = SpeechCommandsDataset(os.path.join('./data', 'speech_commands/speech_commands_v0.01'),
                                              transform=data_transform)
            train_set = build_retain_sets_sc(train_set, num_classes=12)

        if os.path.exists(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(train_set), int(len(train_set) * self.ratio),
                                                replace=False)  #5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np', training_indices)

        source_data = Subset(train_set, training_indices)

        trigger_dataset = []
        for i in range(len(source_data)):
            trigger_dataset.append((source_data[i][0], source_data[i][1]))  # self.target_class
        # for i in range(len(source_data)):
        #     trigger_dataset.append((source_data[i], 0))  # self.target_class
        trigger_loader = DataLoader(trigger_dataset, batch_size=self.batch_size, shuffle=True)  # self.batch_size,
        # target_dataset = []
        # for i in range(len(target_data)):
        # target_dataset.append((target_data[i], 0))  # self.target_class

        for batch_idx, (trigger_samples, targets1) in tqdm(enumerate(trigger_loader)):
            target_class = random.randint(0, self.num_classes - 1)
            trigger_samples = trigger_samples.cuda().requires_grad_()
            output = self.model(trigger_samples)  # inputs
            print(target_class)
            prediction = torch.unbind(output, dim=1)[target_class]
            gradient = torch.autograd.grad(prediction, trigger_samples,
                                           grad_outputs=torch.ones_like(prediction))[0]  # inputs
            trigger_samples = torch.clip(
                trigger_samples + w_lr * torch.sign(gradient[:len(trigger_samples)]),
                0, 1)

            for i in range(len(trigger_samples)):
                img = trigger_samples[i]
                label = targets1[i]
                # unlearninglabels = list(set(range(1, self.num_classes))-set([targets1.cpu().numpy()[i]]))
                # relabel = random.choice(unlearninglabels)
                self.watermark_set.append((img.detach().cpu(), label.numpy()))

            # for i in range(len(trigger_samples)):
            #     img = trigger_samples[i]
            #     unnormalize = transforms.Normalize(mean=[-m / s for m, s in zip(CIFAR_MEAN, CIFAR_STD)],
            #                                        std=[1 / s for s in CIFAR_STD])
            #     img = unnormalize(img)
            #     img_file_name = '%d.png' % (i + batch_idx * batch_size)
            #     img_file_path = os.path.join(self.path, img_file_name)
            #     save_image(img, img_file_path)
            #     print('[Generate Poisoned Set] Save %s' % img_file_path)

            if batch_idx == 0:
                label_set = targets1.cpu().numpy().tolist()
            else:
                label_set += targets1.cpu().numpy().tolist()

            # TODO the length of watermark data
            if len(label_set) > 500:  #500 for cifar10 100 for ?
                break
        # torch.save(torch.LongTensor(label_set), supervisor.get_poison_set_dir(self.args)+'\\labels8-0_wo_snnl')

    #neural cleanse
    def remask(self, label: int, data_loader):
        epoch = self.epoch
        # no bound
        atanh_mark = torch.randn(self.shape, device=self.device)
        atanh_mark.requires_grad_()
        atanh_mask = torch.randn(self.shape[1:], device=self.device)
        atanh_mask.requires_grad_()
        mask = tanh_func(atanh_mask)  # (h, w)
        mark = tanh_func(atanh_mark)  # (c, h, w)

        optimizer = torch.optim.Adam(
            [atanh_mark, atanh_mask], lr=0.1, betas=(0.5, 0.9))
        optimizer.zero_grad()

        cost = self.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        norm_best = float('inf')
        mask_best = None
        mark_best = None
        entropy_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_norm_best = norm_best

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')

        for _epoch in range(epoch):
            satisfy_threshold = False
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            epoch_start = time.time()
            for _input, _label in data_loader:
                _input = self.denormalizer(_input.to(device=self.device))
                _label = _label.to(device=self.device)
                batch_size = _label.size(0)
                X = (_input + mask * (mark - _input)).clamp(0., 1.)
                Y = label * torch.ones_like(_label, dtype=torch.long)
                _output = self.model(self.normalizer(X))

                batch_acc = Y.eq(_output.argmax(1)).float().mean()
                batch_entropy = self.loss_fn(_input, _label, Y, mask, mark, label)
                batch_norm = mask.norm(p=1)
                batch_loss = batch_entropy + cost * batch_norm  # NC loss function

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)  # (h, w)
                mark = tanh_func(atanh_mark)  # (c, h, w)
            epoch_time = time.time() - epoch_start

            if _epoch % 9 == 0:
                pre_str = 'Epoch: {}/{}'.format(_epoch + 1, epoch)
                _str = ' '.join([
                    f'Loss: {losses.avg:.4f},'.ljust(20),
                    f'Acc: {acc.avg:.4f}, '.ljust(20),
                    f'Norm: {norm.avg:.4f},'.ljust(20),
                    f'Entropy: {entropy.avg:.4f},'.ljust(20),
                    f'Time: {epoch_time},'.ljust(20),
                ])

                print(pre_str, _str)

            # check to save best mask or not
            if acc.avg >= self.attack_succ_threshold and (norm.avg < norm_best or satisfy_threshold == False):
                satisfy_threshold = True
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if norm_best < float('inf'):
                    if norm_best >= self.early_stop_threshold * early_stop_norm_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_norm_best = min(norm_best, early_stop_norm_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience:
                    print('early stop')
                    break

            # check cost modification
            if cost == 0 and acc.avg >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    cost = self.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2f' % cost)
            else:
                cost_set_counter = 0

            if acc.avg >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                print('up cost from %.4f to %.4f' % (cost, cost * self.cost_multiplier_up))
                cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                print('down cost from %.4f to %.4f' % (cost, cost / self.cost_multiplier_down))
                cost /= self.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                if acc.avg >= self.attack_succ_threshold: satisfy_threshold = True
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg
        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        return mark_best, mask_best, entropy_best

    def loss_fn(self, _input, _label, Y, mask, mark, label):
        X = (_input + mask * (mark - _input)).clamp(0., 1.)
        Y = label * torch.ones_like(_label, dtype=torch.long)
        _output = self.model(self.normalizer(X))
        return self.criterion(_output, Y)
