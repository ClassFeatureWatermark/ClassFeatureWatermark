"""
Implementation of NAD defense
[1] Li, Yige, et al. "Neural attention distillation: Erasing backdoor triggers from deep neural networks." arXiv preprint arXiv:2101.05930 (2021).
"""
import copy
import logging
import random
import time

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


# do the distillation
class MixRemoval(BackdoorDefense):
    """
    Neural Attention Distillation

    Args:
        teacher_epochs (int): the number of finetuning epochs for a teacher model. Default: 10.
        erase_epochs (int): the number of epochs for erasing the poisoned student model via neural attention distillation. Default: 20.

    .. _Neural Attention Distillation:
        https://openreview.net/pdf?id=9l0K4OM-oXE

    .. _original source code:
        https://github.com/bboylyg/NAD
    """

    def __init__(self, args, teacher_epochs=10, erase_epochs=20):
        super().__init__(args)
        # self.set_seeds()
        self.args = args
        with open(os.path.join(supervisor.get_poison_set_dir(self.args), 'MR_config.yml'), 'r') as file:
            model_config = yaml.safe_load(file)
        self.mr_config = model_config['models'].get(self.args.model[:-3])

        self.teacher_epochs = teacher_epochs
        self.erase_epochs = erase_epochs
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.p = 2  # power for AT
        self.batch_size = 64
        self.betas = np.asarray([500, 1000, 1000])  # hyperparams `betas` for AT loss (for ResNet and WideResNet archs)
        self.threshold_clean = 70.0  # don't save if clean acc drops too much
        # clp
        self.u = self.mr_config['u'] #3.8 for EWE extracted model #3.5 for bandnet
        # rnp
        self.unlearning_lr = self.mr_config['unlearning_lr'] #0.005 for badnet  # 0.01
        self.unlearning_epochs = 20
        self.schedule = [10, 20]
        self.clean_threshold = 0.25  # training acc is evaluated with this threshold
        #adv training
        self.epsilon = self.mr_config['epsilon'] #0.01 for EWE
        self.rnp_factor = self.mr_config['rnp_factor']
        # nad
        self.kd_lr = self.mr_config['kd_lr']  # 0.001

        self.folder_path = 'other_defenses_tool_box/results/MixRemoval'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        # 5% of the clean train set
        if args.dataset == 'cifar10':
            # used for training the teacher
            self.lr = self.mr_config['lr'] #0.002 for Adam #0.1 for SGD
            self.ratio = 0.05  # 0.05 # ratio of training data to use
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'),
                                              train=True, download=True)
            self.data_transform_aug = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  
                transforms.RandomHorizontalFlip(),  
                transforms.RandomRotation(15),  
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
            ])

        elif args.dataset == 'cifar20':
            self.lr = self.mr_config['lr']  # 0.002 for Adam #0.1 for SGD
            self.ratio = 0.05  # 0.05 # ratio of training data to use
            full_train_set = CIFAR20(root=os.path.join(config.data_dir, 'cifar100'),
                                              train=True, download=True)

        elif args.dataset == 'speech_commands':
            self.lr = self.mr_config['lr']  # 0.002 for Adam #0.1 for SGD
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
            full_train_set = build_retain_sets_sc(train_dataset, num_classes=12)

        elif args.dataset == 'gtsrb':
            self.lr = 0.02
            self.ratio = 0.2  # ratio of training data to use
            full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True)
        else:
            raise NotImplementedError()

        self.train_data = DatasetCL(self.ratio, full_dataset=full_train_set, transform=self.data_transform_aug, dataset=self.args.dataset)
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
                                 f'output_MixRemoval_{self.args.model[:-3]}.log')),
                logging.StreamHandler()
            ])
        self.logger.info(self.args)
        print("Initialization finished")

    def set_seeds(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        np.random.seed(seed)
        random.seed(seed)

    def cleanser(self):
        print('Cleaner is running')
        if self.mr_config['run_teacher']:
            self.train_teacher()
        # self.train_rnp_teacher()
        self.train_erase()

    def train_teacher(self):
        """
        Finetune the poisoned model with 5% of the clean train set to obtain a teacher model
        """
        # Load models
        teacher = copy.deepcopy(self.model)
        print('finished teacher model init...')
        self.logger.info('----------- Train Teacher Model (CLP) --------------')

        self.logger.info('Time \t PoisonACC \t CleanACC')
        start = time.time()
        channel_lipschitzness_prune(teacher, self.u)
        end = time.time()
        clean_acc, asr_source, _ = val_atk(self.args, teacher)
        self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
            end - start, asr_source, clean_acc))

        # initialize optimizer
        optimizer = torch.optim.SGD(teacher.parameters(),
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)
        optimizer = torch.optim.Adam(teacher.parameters(),
                                     lr=self.lr,
                                     weight_decay=1e-4)  # SGD
        # define loss functions
        criterion = nn.CrossEntropyLoss().cuda()

        self.logger.info('----------- Teacher in NAD --------------')
        self.logger.info('Time \t PoisonACC \t CleanACC')
        teacher.train()
        for epoch in range(0, self.teacher_epochs):
            start_time = time.time()
            self.adjust_learning_rate(optimizer, epoch, self.lr) #remove

            if epoch == 0:
                print("Test teacher")
                clean_acc, asr, _ = val_atk(self.args, teacher)
                self.logger.info('- \t {:.4f} \t {:.4f}'.format(clean_acc, asr))

            self.train_step(self.train_loader, teacher, optimizer, criterion, epoch + 1)
            end_time = time.time()

            clean_acc, asr_source, _ = val_atk(self.args, teacher)
            self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
                start_time - end_time, asr_source, clean_acc))

        t_model_path = os.path.join(supervisor.get_poison_set_dir(self.args),
                                    f'MR_CN_{self.args.model[:-3]}.pt')
        self.save_checkpoint(teacher.state_dict(), True, t_model_path)

    def train_rnp_teacher(self):
        """
        Finetune the poisoned model with 5% of the clean train set to obtain a rnp_unlearned teacher model
        """
        # Load models
        teacher = copy.deepcopy(self.model)
        teacher.train()
        print("train the rnp teacher")
        val_atk(self.args, teacher)

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(teacher.parameters(), lr=self.unlearning_lr, momentum=0.9,
                                    weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.schedule,
                                                         gamma=0.1)
        bad_t_model_path = os.path.join(supervisor.get_poison_set_dir(self.args),
                                        f'MR_RNP_{self.args.model[:-3]}.pt')

        self.logger.info('----------- Model Unlearning --------------')
        self.logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonACC \t CleanACC')
        for epoch in range(0, self.unlearning_epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_step_unlearning(model=teacher, criterion=criterion,
                                                          optimizer=optimizer,
                                                          data_loader=self.train_loader,
                                                          device=self.device)
            clean_acc, asr_source, _ = val_atk(self.args, teacher)
            scheduler.step()
            end = time.time()
            self.logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                epoch, lr, end - start, train_loss, train_acc, asr_source,
                clean_acc))

            if train_acc <= self.clean_threshold:
                torch.save(teacher.state_dict(), bad_t_model_path)
                break

        # self.save_checkpoint(teacher.state_dict(), True, bad_t_model_path)

    def train_erase(self):
        """
        Erase the backdoor: teach the student (poisoned) model with the teacher model following NAD loss
        """
        self.logger.info('----------- Train Erase --------------')
        self.logger.info('Time \t PoisonACC \t CleanACC')

        # Load models
        arch = config.arch[self.args.dataset]

        teacher = arch(num_classes=self.num_classes)
        bad_teacher = arch(num_classes=self.num_classes)
        # t_model_path = os.path.join(self.folder_path,
        #                             'NAD_T_%s.pt' % supervisor.get_dir_core(self.args, include_model_name=True,
        #                                                                     include_poison_seed=config.record_poison_seed))
        t_model_path = os.path.join(supervisor.get_poison_set_dir(self.args),
                                    f'MR_CN_{self.args.model[:-3]}.pt')
        teacher.load_state_dict(torch.load(t_model_path))

        # bad_t_model_path = os.path.join(supervisor.get_poison_set_dir(self.args),
        #                                 f'MR_RNP_{self.args.model[:-3]}.pt')
        # bad_teacher.load_state_dict(torch.load(bad_t_model_path))

        bad_teacher = bad_teacher.cuda()
        bad_teacher.eval()

        print('test bad teacher model...')
        acc, asr, _ = val_atk(self.args, bad_teacher)
        self.logger.info(f'test bad teacher model\t Clean Acc: \t{acc}\tASR: \t{asr}')

        model = arch(num_classes=self.num_classes)
        t_model_path = supervisor.get_model_dir(self.args)
        checkpoint = torch.load(t_model_path)
        model.load_state_dict(checkpoint)

        # version 1
        # student = teacher
        # teacher = model
        # version 2
        student = model
        # version 3
        # student = copy.deepcopy(teacher)

        student = student.cuda()
        teacher = teacher.cuda()
        teacher.eval()

        print('test teacher model...')
        acc, asr, _ = val_atk(self.args, teacher)
        self.logger.info(f'test teacher model\t Clean Acc: \t{acc}\tASR: \t{asr}')

        ini_student = copy.deepcopy(student)
        ini_student.eval()
        student.train()

        print('finished student model init...')

        nets = {'snet': student, 'tnet': teacher, 'btnet': bad_teacher}

        for param in teacher.parameters():
            param.requires_grad = False

        for param in bad_teacher.parameters():
            param.requires_grad = False

        # initialize optimizer
        # optimizer = torch.optim.SGD(student.parameters(),
        #                             lr=self.kd_lr,
        #                             momentum=0.9,
        #                             weight_decay=1e-4,
        #                             nesterov=True)
        optimizer = torch.optim.Adam(student.parameters(),
                                     lr=self.kd_lr,
                                     weight_decay=1e-4)
        self.args.model = 'MR_' + self.args.model
        self.logger.info('optimizer: SGD\t param groups: {:.4f}'.format(optimizer.param_groups[0]['lr']))
        # define loss functions
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(self.p)

        print('----------- Train Initialization --------------')
        best_acc = 0.0
        for epoch in range(0, self.erase_epochs):
            start_time = time.time()

            # self.adjust_learning_rate_erase(optimizer, epoch, self.lr)

            # train every epoch
            criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

            if epoch == 0:
                # before training test firstly
                val_atk(self.args, student)

            self.train_step_erase(self.train_loader, nets, optimizer,
                                  criterions, epoch + 1, ini_student)

            end_time = time.time()
            clean_acc, asr_source, _ = val_atk(self.args, student)
            self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
                end_time - start_time, asr_source, clean_acc))

            if clean_acc > best_acc:
                print("output_dir,", supervisor.get_model_dir(self.args))
                torch.save(student.state_dict(), supervisor.get_model_dir(self.args))
                best_acc = clean_acc

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
            self.threshold_clean = min(acc_bad, self.threshold_clean)

            best_clean_acc = acc_clean
            best_bad_acc = acc_bad

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

    def test(self, model, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            with torch.no_grad():
                output = model(img)
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg, losses.avg]

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img, target = img.cuda(), target.cuda()
            img = img[target != self.target_class]
            target = target[target != self.target_class]
            poison_img, poison_target = self.poison_transform.transform(img, target)

            with torch.no_grad():
                poison_output = model(poison_img)
                loss = criterion(poison_output, poison_target)

            prec1, prec5 = accuracy(poison_output, poison_target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_bd = [top1.avg, top5.avg, losses.avg]

        print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
        print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

        return acc_clean, acc_bd

    def test_erase(self, nets, criterions, betas, epoch):
        """
        Test the student model at erase step
        """
        top1 = AverageMeter()
        top5 = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.eval()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            with torch.no_grad():
                output_s, _, _, _ = snet.forward(img, return_activation=True)

            prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg]

        cls_losses = AverageMeter()
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img, target = img.cuda(), target.cuda()
            img = img[target != self.target_class]
            target = target[target != self.target_class]
            poison_img, poison_target = self.poison_transform.transform(img, target)

            with torch.no_grad():

                output_s, activation1_s, activation2_s, activation3_s = snet.forward(poison_img, return_activation=True)
                _, activation1_t, activation2_t, activation3_t = tnet.forward(poison_img, return_activation=True)

                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * betas[2]
                at2_loss = criterionAT(activation2_s, activation2_t.detach()) * betas[1]
                at1_loss = criterionAT(activation1_s, activation1_t.detach()) * betas[0]
                at_loss = at3_loss + at2_loss + at1_loss
                cls_loss = criterionCls(output_s, poison_target)

            prec1, prec5 = accuracy(output_s, poison_target, topk=(1, 5))
            cls_losses.update(cls_loss.item(), poison_img.size(0))
            at_losses.update(at_loss.item(), poison_img.size(0))
            top1.update(prec1.item(), poison_img.size(0))
            top5.update(prec5.item(), poison_img.size(0))

        acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

        print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
        print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

        return acc_clean, acc_bd

    def train_step(self, train_loader, model, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

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

            adv_img = self.fgsm_attack(model, nn.CrossEntropyLoss(), img, target, self.epsilon)
            output_adv = model(adv_img)
            loss = criterion(output_adv, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch[{0}]: '
              'loss: {losses.avg:.4f}  '
              'prec@1: {top1.avg:.2f}  '
              'prec@5: {top5.avg:.2f}'.format(epoch, losses=losses, top1=top1, top5=top5))

    # FGSM adversarial example generation
    def fgsm_attack(self, model, loss, data, target, epsilon):
        trigger_samples = copy.deepcopy(data)
        trigger_samples.requires_grad = True
        output = model(trigger_samples)
        # for _ in range(5):
        #     output = model(trigger_samples)
        #     prediction = torch.unbind(output, dim=1)[0] #target class
        #     gradient = torch.autograd.grad(prediction, triggers_samples,
        #                                    grad_outputs=torch.ones_like(prediction))[0]
        #     trigger_samples = torch.clip(
        #         trigger_samples + epsilon * torch.sign(gradient[:len(trigger_samples)]),
        #         0, 1)
        model.zero_grad()
        cost = loss(output, target).to(trigger_samples.device)
        cost.backward()
        trigger_samples = trigger_samples + epsilon * trigger_samples.grad.sign()
        trigger_samples = torch.clamp(trigger_samples, 0, 1)
        return trigger_samples
    def train_step_erase(self, train_loader, nets, optimizer, criterions, epoch, ini_student):
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']
        btnet = nets['btnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            if self.args.dataset == 'speech_commands':
                output_s, hidden_s = snet.forward(img, return_hidden=True)
                _, hidden_t = tnet.forward(img, return_hidden=True)
                cls_loss = criterionCls(output_s, target)
                at3_loss = torch.nn.functional.mse_loss(hidden_s.flatten(), hidden_t.flatten().detach()) * self.betas[2]
                at_loss = at3_loss + cls_loss
            else:
                output_s, activation1_s, activation2_s, activation3_s = snet.forward(img, return_activation=True)
                _, activation1_t, activation2_t, activation3_t = tnet.forward(img, return_activation=True)

                # bt_loss = criterionCls(output_s, bt_out)
                # target = torch.argmax(ini_student(img), 1)
                # adversarial training
                # Generate adversarial examples

                # version 1
                # adv_img = self.fgsm_attack(snet, nn.CrossEntropyLoss(), img, target, self.epsilon)
                # output_s_adv = snet(adv_img)
                # bt_out = btnet(adv_img)
                # bt_targets = torch.nn.functional.softmax(bt_out, dim=1)#argmax
                # bt_loss = criterionCls(output_s_adv, bt_targets)
                # version 2
                # bt_targets = torch.nn.functional.log_softmax(btnet(img), dim=1)
                bt_targets = torch.argmax(btnet(img), dim=1)
                bt_loss = criterionCls(output_s, bt_targets)

                # version 2
                cls_loss = criterionCls(output_s, target)  # + 0.1*criterionCls(output_s_adv, target)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * self.betas[2]
                at2_loss = criterionAT(activation2_s, activation2_t.detach()) * self.betas[1]
                at1_loss = criterionAT(activation1_s, activation1_t.detach()) * self.betas[0]
                #
                # atb3_loss = criterionAT(activation3_s, activation3_b.detach()) * self.betas[2]
                # atb2_loss = criterionAT(activation2_s, activation2_b.detach()) * self.betas[1]
                # atb1_loss = criterionAT(activation1_s, activation1_b.detach()) * self.betas[0]

                at_loss = (at1_loss + at2_loss + at3_loss + cls_loss)
                           # + self.rnp_factor / bt_loss)  # (atb3_loss + atb2_loss + atb1_loss)

            #version 1
            # target = torch.argmax(tnet(img), dim=1)
            # at_loss = cls_loss - self.rnp_factor * bt_loss

            prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
            at_losses.update(at_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            at_loss.backward()
            optimizer.step()

        print('Epoch[{0}]: '
              'AT_loss: {losses.avg:.4f}  '
              'prec@1: {top1.avg:.2f}  '
              'prec@5: {top5.avg:.2f}'.format(epoch, losses=at_losses, top1=top1, top5=top5))

    def adjust_learning_rate(self, optimizer, epoch, lr):
        # The learning rate is divided by 10 after every 2 epochs
        lr = lr * math.pow(10, -math.floor(epoch / 2))

        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate_erase(self, optimizer, epoch, lr):
        if epoch < 2:
            lr = lr
        elif epoch < 20:
            # lr = 0.01
            lr *= 0.1
        elif epoch < 30:
            # lr = 0.0001
            lr *= 0.001
        else:
            # lr = 0.0001
            lr *= 0.001
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, state, is_best, save_dir):
        if is_best:
            torch.save(state, save_dir)
            print('[info] save best model')


class NAD(BackdoorDefense):
    """
    Neural Attention Distillation

    Args:
        teacher_epochs (int): the number of finetuning epochs for a teacher model. Default: 10.
        erase_epochs (int): the number of epochs for erasing the poisoned student model via neural attention distillation. Default: 20.

    .. _Neural Attention Distillation:
        https://openreview.net/pdf?id=9l0K4OM-oXE


    .. _original source code:
        https://github.com/bboylyg/NAD

    """

    def __init__(self, args, teacher_epochs=10, erase_epochs=20):
        super().__init__(args)

        self.args = args
        self.teacher_epochs = teacher_epochs
        self.erase_epochs = erase_epochs

        self.p = 2  # power for AT
        self.batch_size = 64
        self.betas = [500, 1000, 1000]  # hyperparams `betas` for AT loss (for ResNet and WideResNet archs)
        self.threshold_clean = 70.0  # don't save if clean acc drops too much

        self.folder_path = './other_defenses_tool_box/results/NAD'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

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
        elif args.dataset == 'gtsrb':
            self.lr = 0.02
            self.ratio = 0.5  # ratio of training data to use
            full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True)
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

    def detect(self):
        self.train_teacher()
        self.train_erase()

    def train_teacher(self):
        """
        Finetune the poisoned model with 5% of the clean train set to obtain a teacher model
        """
        # Load models
        print('----------- Network Initialization --------------')
        teacher = copy.deepcopy(self.model)
        teacher.train()
        print('finished teacher model init...')

        # initialize optimizer
        optimizer = torch.optim.SGD(teacher.parameters(),
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)#SGD
        # optimizer = torch.optim.Adam(teacher.parameters(),
        #                              lr=self.lr,
        #                              weight_decay=1e-4)  # SGD
        self.logger.info('teacher optimizer \t {} \t '.format(optimizer.__class__))

        # define loss functions
        criterion = nn.CrossEntropyLoss().cuda()

        print('----------- Train Initialization --------------')
        self.logger.info('----------- Teacher in NAD --------------')
        self.logger.info('Time \t PoisonACC \t CleanACC')
        for epoch in range(0, self.teacher_epochs):
            start_time = time.time()
            self.adjust_learning_rate(optimizer, epoch, self.lr)

            if epoch == 0:
                # before training test firstly
                # self.test(teacher, criterion, epoch)
                # test(teacher, test_loader=self.test_loader,
                #      poison_test=True, poison_transform=self.poison_transform,
                #      num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
                print("Test teacher")
                clean_acc, asr, _ = val_atk(self.args, teacher)
                self.logger.info('- \t {:.4f} \t {:.4f}'.format(clean_acc, asr))

            self.train_step(self.train_loader, teacher, optimizer, criterion, epoch + 1)
            end_time = time.time()

            # evaluate on testing set
            # acc_clean, acc_bad = self.test(teacher, criterion, epoch+1)
            # acc_clean, acc_bad = acc_clean[0], acc_bad[0]
            # acc_clean, acc_bad = test(teacher, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
            # val_atk(self.args, teacher)
            clean_acc, asr_source, _ = val_atk(self.args, teacher)
            self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
                end_time - start_time, asr_source, clean_acc))

            # remember best precision and save checkpoint
            # is_best = acc_clean[0] > self.threshold_clean
            # self.threshold_clean = min(acc_bad, self.threshold_clean)

            # best_clean_acc = acc_clean
            # best_bad_acc = acc_bad

            t_model_path = os.path.join(self.folder_path, 'NAD_T_%s.pt' % supervisor.get_dir_core(self.args,
                                                                                                  include_model_name=True,
                                                                                                  include_poison_seed=config.record_poison_seed))
            self.save_checkpoint(teacher.state_dict(), True, t_model_path)

    def train_erase(self):
        """
        Erase the backdoor: teach the student (poisoned) model with the teacher model following NAD loss
        """
        # Load models
        print('----------- Network Initialization --------------')
        self.logger.info('----------- Knowledge Distillation in NAD --------------')
        self.logger.info('Time \t PoisonACC \t CleanACC')
        # if self.args.poison_type == 'meadefender':
        #     arch = MEANet
        # else:
        #     arch = config.arch[self.args.dataset]
        # arch = config.arch[self.args.dataset]
        # teacher = arch(num_classes=self.num_classes)
        teacher = copy.deepcopy(self.model)
        t_model_path = os.path.join(self.folder_path,
                                    'NAD_T_%s.pt' % supervisor.get_dir_core(self.args, include_model_name=True,
                                                                            include_poison_seed=config.record_poison_seed))
        checkpoint = torch.load(t_model_path)
        teacher.load_state_dict(checkpoint)
        teacher = teacher.cuda()
        teacher.eval()
        print("evaluate teacher:")
        val_atk(self.args, teacher)

        # student = arch(num_classes=self.num_classes)
        # s_model_path = supervisor.get_model_dir(self.args)
        # checkpoint = torch.load(s_model_path)
        # student.load_state_dict(checkpoint)
        # student = student.cuda()

        student = copy.deepcopy(self.model)
        student.train()

        print('finished student model init...')

        nets = {'snet': student, 'tnet': teacher}

        for param in teacher.parameters():
            param.requires_grad = False

        # initialize optimizer
        optimizer = torch.optim.SGD(student.parameters(),
                                    lr=0.01,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)

        self.logger.info('optimizer: SGD\t param groups: {:.4f}'.format(optimizer.param_groups[0]['lr']))
        # define loss functions
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(self.p)

        print('----------- Train Initialization --------------')
        for epoch in range(0, self.erase_epochs):
            start_time = time.time()

            # self.adjust_learning_rate_erase(optimizer, epoch, self.lr)

            # train every epoch
            criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

            if epoch == 0:
                # before training test firstly
                # self.test_erase(nets, criterions, self.betas, epoch)
                # test(student, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
                val_atk(self.args, student)

            self.train_step_erase(self.train_loader, nets, optimizer, criterions,
                                  self.betas, epoch + 1)

            end_time = time.time()
            # evaluate on testing set
            # acc_clean, acc_bad = self.test_erase(nets, criterions, self.betas, epoch+1)
            # acc_clean, acc_bad = acc_clean[0], acc_bad[0]
            # acc_clean, acc_bad = test(student, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
            acc_clean = val_atk(self.args, student)[0]

            # remember best precision and save checkpoint
            # is_best = acc_clean > self.threshold_clean
            # self.threshold_clean = min(acc_bad, self.threshold_clean)

            # best_clean_acc = acc_clean
            # best_bad_acc = acc_bad

            clean_acc, asr_source, _ = val_atk(self.args, student)
            self.logger.info('{:.1f} \t {:.4f} \t {:.4f}'.format(
                start_time - end_time, asr_source, clean_acc))

            # erase_model_path = os.path.join(self.folder_path, 'NAD_E_%s.pt' % supervisor.get_dir_core(self.args,
            #                                                                                           include_model_name=True,
            #                                                                                           include_poison_seed=config.record_poison_seed))
            # self.save_checkpoint(student.state_dict(), is_best, erase_model_path)

        self.args.model = 'NAD_' + self.args.model
        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(student.state_dict(), supervisor.get_model_dir(self.args))

    # def test(self, model, criterion, epoch):
    #     losses = AverageMeter()
    #     top1 = AverageMeter()
    #     top5 = AverageMeter()
    #
    #     model.eval()
    #
    #     for idx, (img, target) in enumerate(self.test_loader, start=1):
    #         img = img.cuda()
    #         target = target.cuda()
    #
    #         with torch.no_grad():
    #             output = model(img)
    #             loss = criterion(output, target)
    #
    #         prec1, prec5 = accuracy(output, target, topk=(1, 5))
    #         losses.update(loss.item(), img.size(0))
    #         top1.update(prec1.item(), img.size(0))
    #         top5.update(prec5.item(), img.size(0))
    #
    #     acc_clean = [top1.avg, top5.avg, losses.avg]
    #
    #     losses = AverageMeter()
    #     top1 = AverageMeter()
    #     top5 = AverageMeter()
    #
    #     for idx, (img, target) in enumerate(self.test_loader, start=1):
    #         img, target = img.cuda(), target.cuda()
    #         img = img[target != self.target_class]
    #         target = target[target != self.target_class]
    #         poison_img, poison_target = self.poison_transform.transform(img, target)
    #
    #         with torch.no_grad():
    #             poison_output = model(poison_img)
    #             loss = criterion(poison_output, poison_target)
    #
    #         prec1, prec5 = accuracy(poison_output, poison_target, topk=(1, 5))
    #         losses.update(loss.item(), img.size(0))
    #         top1.update(prec1.item(), img.size(0))
    #         top5.update(prec5.item(), img.size(0))
    #
    #     acc_bd = [top1.avg, top5.avg, losses.avg]
    #
    #     print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
    #     print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))
    #
    #     return acc_clean, acc_bd

    def test_erase(self, nets, criterions, betas, epoch):
        """
        Test the student model at erase step
        """
        top1 = AverageMeter()
        top5 = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.eval()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            with torch.no_grad():
                output_s, _, _, _ = snet.forward(img, return_activation=True)

            prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg]

        cls_losses = AverageMeter()
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img, target = img.cuda(), target.cuda()
            img = img[target != self.target_class]
            target = target[target != self.target_class]
            poison_img, poison_target = self.poison_transform.transform(img, target)

            with torch.no_grad():
                output_s, activation1_s, activation2_s, activation3_s = snet.forward(poison_img, return_activation=True)
                _, activation1_t, activation2_t, activation3_t = tnet.forward(poison_img, return_activation=True)

                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * betas[2]
                at2_loss = criterionAT(activation2_s, activation2_t.detach()) * betas[1]
                at1_loss = criterionAT(activation1_s, activation1_t.detach()) * betas[0]
                at_loss = at3_loss + at2_loss + at1_loss
                cls_loss = criterionCls(output_s, poison_target)

            prec1, prec5 = accuracy(output_s, poison_target, topk=(1, 5))
            cls_losses.update(cls_loss.item(), poison_img.size(0))
            at_losses.update(at_loss.item(), poison_img.size(0))
            top1.update(prec1.item(), poison_img.size(0))
            top5.update(prec5.item(), poison_img.size(0))

        acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

        print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
        print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

        return acc_clean, acc_bd

    def train_step(self, train_loader, model, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            output = model(img)

            loss = criterion(output, target)

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

    def train_step_erase(self, train_loader, nets, optimizer, criterions, betas, epoch):
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            output_s, activation1_s, activation2_s, activation3_s = snet.forward(img, return_activation=True)
            _, activation1_t, activation2_t, activation3_t = tnet.forward(img, return_activation=True)

            cls_loss = criterionCls(output_s, target)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * self.betas[2]
            at2_loss = criterionAT(activation2_s, activation2_t.detach()) * self.betas[1]
            at1_loss = criterionAT(activation1_s, activation1_t.detach()) * self.betas[0]
            if self.args.poison_type == 'meadefender':
                at_loss = cls_loss + 0.01 * at1_loss
            else:
                at_loss = at1_loss + at2_loss + at3_loss + cls_loss

            prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
            at_losses.update(at_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            at_loss.backward()
            optimizer.step()

        print('Epoch[{0}]: '
              'AT_loss: {losses.avg:.4f}  '
              'prec@1: {top1.avg:.2f}  '
              'prec@5: {top5.avg:.2f}'.format(epoch, losses=at_losses, top1=top1, top5=top5))

    def adjust_learning_rate(self, optimizer, epoch, lr):
        # The learning rate is divided by 10 after every 2 epochs
        lr = lr * math.pow(10, -math.floor(epoch / 2))

        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate_erase(self, optimizer, epoch, lr):
        if epoch < 2:
            lr = lr
        elif epoch < 20:
            # lr = 0.01
            lr *= 0.1
        elif epoch < 30:
            # lr = 0.0001
            lr *= 0.001
        else:
            # lr = 0.0001
            lr *= 0.001
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, state, is_best, save_dir):
        if is_best:
            torch.save(state, save_dir)
            print('[info] save best model')


'''
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''


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
