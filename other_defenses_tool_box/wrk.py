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

class WRK(BackdoorDefense):

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

        # clp
        self.u = 3.4

        self.p = 2  # power for AT
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
        self.adv_vulnerability()
        self.neural_cleanse(load=False)

        if not self.has_trigger:
           self.adversarial_sample_set() #MEAdefender, EWE, WBM

        self.relabel() #unlearning

        if not self.has_trigger and not self.adversarial_vulnerable:
            self.args.model = 'WMR_' + self.args.model[:-3] + f'_lr{self.lr}'+self.args.model[-3:]
            self.data_aug_finetuning() #MEADefender, or WBM
        else:
            self.args.model = 'WMR_' + self.args.model

        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))

    def neural_cleanse(self, load):
        self.logger.info('----neural cleanse detection----')
        #step 1, reverse possible
        #no aug dataset
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)
        ])
        train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        source_data = Subset(train_set, list(range(2500)))
        dataloader = DataLoader(source_data, batch_size=640)#512

        if load == False:
            mark_list, mask_list, loss_list = [], [], []
            for label in range(self.num_classes):
                # for label in [self.target_class]:
                print('Class: %d/%d' % (label + 1, self.num_classes))
                mark, mask, loss = self.remask(label, dataloader)
                mark_list.append(mark)
                mask_list.append(mask)
                loss_list.append(loss)
            mark_list = torch.stack(mark_list)
            mask_list = torch.stack(mask_list)
            loss_list = torch.as_tensor(loss_list)
            file_path = os.path.normpath(os.path.join(
                'other_defenses_tool_box/results/NC', 'neural_cleanse_%s.npz' % supervisor.get_dir_core(self.args,
                                                                                                        include_model_name=True,
                                                                                                        include_poison_seed=config.record_poison_seed)))
            np.savez(file_path, mark_list=[to_numpy(mark) for mark in mark_list],
                     mask_list=[to_numpy(mask) for mask in mask_list],
                     loss_list=loss_list)
        else:
            file_path = os.path.normpath(os.path.join(
                'other_defenses_tool_box/results/NC', 'neural_cleanse_%s.npz' % supervisor.get_dir_core(self.args,
                                                                                    include_model_name=True,
                                                                                    include_poison_seed=config.record_poison_seed)))
            f = np.load(file_path)
            mark_list, mask_list, loss_list = torch.tensor(f['mark_list']), torch.tensor(f['mask_list']), torch.tensor(
                f['loss_list'])

        mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1)
        print('mask norms: ', mask_norms)
        print('mask anomaly indices: ', normalize_mad(mask_norms))
        print('loss: ', loss_list)
        print('loss anomaly indices: ', normalize_mad(loss_list))

        anomaly_indices = normalize_mad(mask_norms)
        suspect_classes = []
        suspect_classes_anomaly_indices = []
        for i in range(self.num_classes):
            if mask_norms[i] > torch.median(mask_norms): continue
            if anomaly_indices[i] > self.has_trigger_threshold:
                suspect_classes.append(i)
                suspect_classes_anomaly_indices.append(anomaly_indices[i])

        print("Suspect Classes:", suspect_classes)
        if len(suspect_classes) > 0:
            self.has_trigger = True
            max_idx = torch.tensor(suspect_classes_anomaly_indices).argmax().item()
            suspect_class = suspect_classes[max_idx]
            mask = mask_list[suspect_class]
            mark = mark_list[suspect_class]
            self.mask = mask.cpu()
            self.mark = mark.cpu()
            self.watermark_set = []

            train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                         download=True, transform=transforms.ToTensor())

            source_data = Subset(train_set, list(range(2500)))
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])

            for img, label in Subset(source_data, list(range(1000))):
                img = img * (1 - self.mask) + self.mark * self.mask
                img = data_transform_aug(img)
                self.watermark_set.append((img, label))

    def adv_vulnerability(self):
        def fgsm_attack(model, loss_fn, image, label, epsilon):

            image.requires_grad = True

            output = model(image)
            loss = loss_fn(output, label)
            model.zero_grad()
            loss.backward()

            perturbation = epsilon * image.grad.data.sign()

            adversarial_image = image + perturbation
            adversarial_image = torch.clamp(adversarial_image, 0, 1)  # 保证像素值在有效范围内

            return adversarial_image, perturbation

        def find_minimal_distance(model, loss_fn, data, label, max_epsilon=3.0, step_size=0.01):
            current_epsilon = 0
            data = data.clone().detach()
            output = model(data)
            # print("output.shape", output.shape)
            initial_prediction = output.max(1, keepdim=True)[1]
            # print("initial_prediction", initial_prediction)

            if initial_prediction.item() != label.item():
                return "misclassified_data"
                #raise ValueError("Initial image is already misclassified.")

            while current_epsilon <= max_epsilon:
                adversarial_image, perturbation = fgsm_attack(model, loss_fn, data, label, current_epsilon)

                output = model(adversarial_image)
                final_prediction = output.max(1, keepdim=True)[1]

                if final_prediction.item() != label.item():
                    l2_distance = torch.norm(perturbation.view(perturbation.size(0), -1), dim=1).item()
                    return l2_distance

                current_epsilon += step_size
            return 0.0
        #source_data, no aug
        if self.args.dataset =='cifar10':
            CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
            train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                         download=True, transform=data_transform)
        else:
            train_set = self.train_data
        source_data = Subset(train_set, list(range(100)))
        dataloader = DataLoader(source_data, batch_size=1, shuffle=False)

        self.model.eval()

        loss_fn = nn.CrossEntropyLoss()
        min_distance = 0.0
        total_len = len(source_data)
        for batch in tqdm(dataloader):
            data = batch[0]
            label = batch[1]
            data = data.to(self.device)
            label = torch.tensor(label).to(self.device)
            min_dis = find_minimal_distance(self.model, loss_fn, data, label)
            if min_dis == 'misclassified_data':
                total_len -= 1
            else:
                min_distance += min_dis

        print("min_distance/len(source_data)",  min_distance/total_len) #5.65->badnet, 2.01->ewe, 4.77->meadefender, 2.67->MBW

        if min_distance/total_len < self.adversarial_vulnerable_threshold:
            self.adversarial_vulnerable = True

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
                target=torch.argmax(original_model(img), dim=1)

            output = model(img)
            # output_adv = model(adv_img)

            loss = criterion(output, target) #+ criterion(output_adv, target)

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

    def data_aug_finetuning(self):
        if self.args.dataset == 'cifar10':
            #TODO
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(15),  # this
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # this
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

    def relabel(self):
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
            unlearning_trainset, 128, pin_memory=True, shuffle=True
        )

        criterion = nn.CrossEntropyLoss().cuda()
        if self.args.dataset == 'speech_commands':
            lr = self.wmr_config['relabel_lr']
        else:
            lr = 0.01
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
        batch_size = 128
        w_lr = 0.15 #0.15

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

        if os.path.exists(supervisor.get_poison_set_dir(self.args)+'/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(self.args)+'/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(train_set), 2500, replace=False) #5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args)+'/domain_set_indices.np', training_indices)

        source_data = Subset(train_set, training_indices)

        trigger_dataset = []
        for i in range(len(source_data)):
            trigger_dataset.append((source_data[i][0], source_data[i][1]))  # self.target_class
        # for i in range(len(source_data)):
        #     trigger_dataset.append((source_data[i], 0))  # self.target_class
        trigger_loader = DataLoader(trigger_dataset, batch_size=batch_size, shuffle=True)  # self.batch_size,
        # target_dataset = []
        # for i in range(len(target_data)):
            # target_dataset.append((target_data[i], 0))  # self.target_class

        for batch_idx, (trigger_samples, targets1) in tqdm(enumerate(trigger_loader)):
            target_class = random.randint(0, self.num_classes-1)
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
                unlearninglabels = list(set(range(1, self.num_classes))-set([targets1.cpu().numpy()[i]]))
                relabel = random.choice(unlearninglabels)
                self.watermark_set.append((img.detach().cpu(), relabel))

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
            if len(label_set) > 500:#500 for cifar10 100 for ?
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
        mask = tanh_func(atanh_mask)    # (h, w)
        mark = tanh_func(atanh_mark)    # (c, h, w)

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
                batch_loss = batch_entropy + cost * batch_norm # NC loss function

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)    # (h, w)
                mark = tanh_func(atanh_mark)    # (c, h, w)
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

    # def blindspot(self):
    #     # self.train_loader is retain loader
    #     clean_acc, asr_source, _ = val_atk(self.args, self.model)
    #     self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
    #         0.00, asr_source, clean_acc))
    #
    #     data_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)
    #     ])
        # poison_set_dir = supervisor.get_poison_set_dir(self.args)
        # poisoned_set_img_dir = os.path.join(poison_set_dir, 'data8-0_wo_snnl')#data; data9-0
        # poisoned_set_label_path = os.path.join(poison_set_dir, 'labels8-0_wo_snnl')#labels; labels9-0
        # print('dataset : %s' % poisoned_set_img_dir)
        # poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
        #                                  label_path=poisoned_set_label_path,
        #                                  transforms=data_transform)  # if args.no_aug else data_transform_aug

        # arch = config.arch[self.args.dataset]
        # unlearning_teacher = arch(num_classes=(10))
        # t_model_path = os.path.join(supervisor.get_poison_set_dir(self.args),
        #                             f'MR_CN_{self.args.model[:-3]}.pt')
        # unlearning_teacher.load_state_dict(torch.load(t_model_path))
        # unlearning_teacher = nn.DataParallel(unlearning_teacher)
        # unlearning_teacher = unlearning_teacher.cuda()

        # teacher_model = copy.deepcopy(self.model)
        # KL_temperature = 1.0
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)  #0.00002 for previous unlearning

        # blindspot_unlearner(
        #     model=self.model,
        #     unlearning_teacher=unlearning_teacher,
        #     full_trained_teacher=teacher_model,
        #     retain_data=self.train_data,
        #     forget_data=poisoned_set,#Subset(poisoned_set, np.random.choice(list(range(len(poisoned_set))), size=500)),
        #     epochs=15,
        #     optimizer=optimizer,
        #     batch_size=256,
        #     device=self.device,
        #     KL_temperature=KL_temperature,
        # )

    # def create_poison_set(self):
    #     arch = config.arch[self.args.dataset]
    #     #here we have a unlearning teacher which has 0 accuracy;
    #     #if we
    #     unlearning_teacher = arch(num_classes=(10))
    #     un_model_path = os.path.join(supervisor.get_poison_set_dir(self.args),
    #                                  f'full_base_aug_seed=2333-step1-100.pt')
    #                                 # f'MR_CN_full_base_aug_seed=2333.pt')
    #     unlearning_teacher.load_state_dict(torch.load(un_model_path))
    #     unlearning_teacher = nn.DataParallel(unlearning_teacher)
    #     unlearning_teacher = unlearning_teacher.cuda()
    #     unlearning_teacher.eval()
    #
    #     acc, asr, _ = val_atk(self.args, unlearning_teacher)
    #     print('test unlearning teacher', acc, ',', asr)
    #
    #     # acc, asr, _ = val_atk(self.args, self.model)
    #     # print('test self.model ', acc, ',', asr)
    #
    #     # trigger_loader = DataLoader(self.trigger_set, batch_size=self.batch_size, shuffle=True)
    #     print("fgsm_optimze_trigger")
    #     classwise_train_data = get_classwise_ds(self.full_train_data2, num_classes=self.num_classes)
    #
    #     # self.watermark_set = []
    #     # watermark_set_lists = {}
    #     # for i in range(self.num_classes):
    #     #     watermark_set_lists[i] = []
    #     # target_class = 0
    #     w_lr = 0.01#0.01
    #     # loss = nn.CrossEntropyLoss()
    #     # difference_distance_matrix = np.zeros([self.num_classes, self.num_classes])#target, source
    #     for target_class in range(1):
    #         # target_data = classwise_train_data[int(target_class)]  # 8
    #         # target_loader = DataLoader(target_data, batch_size=256, shuffle=True)
    #         # iter_target_loader = iter(target_loader)
    #         source_data = classwise_train_data[8]  # 8
    #         trigger_loader = DataLoader(source_data, batch_size=128, shuffle=False)
    #
    #         for batch_idx, (trigger_samples, targets) in tqdm(enumerate(trigger_loader)):
    #             for epoch in range(10):
    #                 trigger_samples = trigger_samples.cuda().requires_grad_()
    #                 output = unlearning_teacher(trigger_samples)  # ewe/activation
    #                 prediction = torch.unbind(output, dim=1)[target_class]
    #                 gradient = \
    #                     torch.autograd.grad(prediction, trigger_samples, grad_outputs=torch.ones_like(prediction))[
    #                         0]
    #                 trigger_samples = torch.clip(
    #                     trigger_samples - w_lr * torch.sign(gradient[:len(trigger_samples)]),
    #                     0, 1)
    #
    #                 for _ in range(5):
    #                     # inputs = torch.vstack((trigger_samples, trigger_samples))
    #                     output = unlearning_teacher(trigger_samples)
    #                     prediction = torch.unbind(output, dim=1)[target_class]
    #                     gradient = torch.autograd.grad(prediction, trigger_samples,
    #                                                    grad_outputs=torch.ones_like(prediction))[0]
    #                     trigger_samples = torch.clip(
    #                         trigger_samples + w_lr * torch.sign(gradient[:len(trigger_samples)]),
    #                         0, 1)
    #             # for idx in range(len(trigger_samples)):
    #             #     self.watermark_set.append((trigger_samples[idx].detach().cpu(), targets[idx].numpy()))#trigger_set
    #             for i in range(len(trigger_samples)):
    #                 img = trigger_samples[i]
    #                 unnormalize = transforms.Normalize(mean=[-m / s for m, s in zip(self.CIFAR_MEAN, self.CIFAR_STD)],
    #                                                    std=[1 / s for s in self.CIFAR_STD])
    #                 img = unnormalize(img)
    #                 img_file_name = '%d.png' % (i + batch_idx * self.batch_size)
    #                 img_file_path = os.path.join(supervisor.get_poison_set_dir(self.args)+'\\data8-0_wo_snnl', img_file_name)
    #                 save_image(img, img_file_path)
    #                 print('[Generate Poisoned Set] Save %s' % img_file_path)
    #
    #             if batch_idx == 0:
    #                 label_set = targets.cpu().numpy().tolist()
    #             else:
    #                 label_set += targets.cpu().numpy().tolist()
    #
    #             if batch_idx > 2:
    #                 break
    #
    #         torch.save(torch.LongTensor(label_set), supervisor.get_poison_set_dir(self.args)+'\\labels8-0_wo_snnl')

        # for source_class in list(set(range(10)) - set([target_class])):
            #     source_data = classwise_train_data[int(source_class)]  # 8
            #     trigger_loader = DataLoader(source_data, batch_size=256, shuffle=True)
            #     # trigger_set = []
            #     for batch_idx, (trigger_samples, targets1) in tqdm(enumerate(trigger_loader)):
            #         for epoch in range(10):
            #             trigger_samples = trigger_samples.cuda().requires_grad_()
            #             output = unlearning_teacher(trigger_samples)  # ewe/activation
            #             prediction = torch.unbind(output, dim=1)[target_class]
            #             gradient = \
            #                 torch.autograd.grad(prediction, trigger_samples, grad_outputs=torch.ones_like(prediction))[
            #                     0]
            #             trigger_samples = torch.clip(
            #                 trigger_samples - w_lr * torch.sign(gradient[:len(trigger_samples)]),
            #                 0, 1)
            #
            #             for _ in range(5):
            #                 # inputs = torch.vstack((trigger_samples, trigger_samples))
            #                 output = unlearning_teacher(trigger_samples)
            #                 prediction = torch.unbind(output, dim=1)[target_class]
            #                 gradient = torch.autograd.grad(prediction, trigger_samples,
            #                                                grad_outputs=torch.ones_like(prediction))[0]
            #                 trigger_samples = torch.clip(
            #                     trigger_samples + w_lr * torch.sign(gradient[:len(trigger_samples)]),
            #                     0, 1)
            #         for idx in range(len(trigger_samples)):
            #             self.watermark_set.append((trigger_samples[idx], targets1[idx]))#trigger_set

                #let's test
                # self.model.eval()
                # test_trigger_loader = DataLoader(trigger_set, batch_size=128, shuffle=False)
                # num_equal = 0.0
                # total_num = 0.0
                # for trigger_samples2, _ in test_trigger_loader:
                #     # print("trigger_samples", trigger_samples)
                #     pred1 = torch.argmax(self.model(trigger_samples2), dim=1)
                #     pred2 = torch.argmax(unlearning_teacher(trigger_samples2), dim=1)
                #     num_equal += (pred1 != pred2).sum().item()
                #     total_num += len(trigger_samples2)

                # difference_distance = num_equal / total_num
                # difference_distance_matrix[target_class, source_class] = difference_distance
                # print("Source Label:", source_class, ", Target Label:", target_class, ", difference distance:", difference_distance)
        #
        # CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        # CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        # transform = transforms.Compose([
        #     transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        # ])
        # self.watermark_set = Dataset4Tuple(self.watermark_set, transform)
        # lets test, so that the they are different


class ADV(BackdoorDefense):

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

        # clp
        self.u = 3.4

        self.p = 2  # power for AT
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
                # transforms.RandomRotation(15),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
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
        self.adversarial_sample_set() #MEAdefender, EWE, WBM
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
                target=torch.argmax(original_model(img), dim=1)

            output = model(img)
            # output_adv = model(adv_img)

            loss = criterion(output, target) #+ criterion(output_adv, target)

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
            unlearning_trainset, 128, pin_memory=True, shuffle=True
        )

        criterion = nn.CrossEntropyLoss().cuda()
        if self.args.dataset == 'speech_commands':
            lr = self.wmr_config['relabel_lr']
        else:
            lr = 0.01
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
        batch_size = 128
        w_lr = 0.3 #0.15

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

        if os.path.exists(supervisor.get_poison_set_dir(self.args)+'/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(self.args)+'/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(train_set), 25000, replace=False) #5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args)+'/domain_set_indices.np', training_indices)

        source_data = Subset(train_set, training_indices)

        trigger_dataset = []
        for i in range(len(source_data)):
            trigger_dataset.append((source_data[i][0], source_data[i][1]))  # self.target_class
        # for i in range(len(source_data)):
        #     trigger_dataset.append((source_data[i], 0))  # self.target_class
        trigger_loader = DataLoader(trigger_dataset, batch_size=batch_size, shuffle=True)  # self.batch_size,
        # target_dataset = []
        # for i in range(len(target_data)):
            # target_dataset.append((target_data[i], 0))  # self.target_class

        for batch_idx, (trigger_samples, targets1) in tqdm(enumerate(trigger_loader)):
            target_class = random.randint(0, self.num_classes-1)
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
            if len(label_set) > 500:#500 for cifar10 100 for ?
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
        mask = tanh_func(atanh_mask)    # (h, w)
        mark = tanh_func(atanh_mark)    # (c, h, w)

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
                batch_loss = batch_entropy + cost * batch_norm # NC loss function

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)    # (h, w)
                mark = tanh_func(atanh_mark)    # (c, h, w)
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