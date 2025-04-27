import torch
from torch import nn
import  torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
import numpy as np
from torchvision.datasets import CIFAR100
from torchvision.utils import save_image

import queries
import tools
from EmbedModule import Embbed
from U_Net_Zoo import U_Net
from cifar20 import OodDataset, CIFAR20, OodDataset_from_Dataset, FixedLabelDataset
from entangled_watermark import get_classwise_ds_ewe
from imagenet import Imagenet64
from mea_dataset import MixDataset, com_MixDataset
from mea_mixer import *
from mu_utils import build_retain_sets_in_unlearning, get_classwise_ds, build_retain_sets_sc, build_ood_sets_sc
from speech_commands import SpeechCommandsDataset
from speech_transforms import ToTensor, ToMelSpectrogram, FixAudioLength, LoadAudio, ToMelSpectrogramFromSTFT, \
    DeleteSTFT, TimeshiftAudioOnSTFT, ChangeSpeedAndPitchAudio, ToSTFT, StretchAudioOnSTFT, FixSTFTDimension, \
    ChangeAmplitude

from utils import supervisor
from utils.tools import IMG_Dataset
import config

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str = None, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def to_numpy(x, **kwargs) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, **kwargs)

# Project function
def tanh_func(x: torch.Tensor) -> torch.Tensor:
    return (x.tanh() + 1) * 0.5

def generate_dataloader(dataset='cifar10', dataset_path='./data/', batch_size=128, split='train',
                        shuffle=True, drop_last=False, data_transform=None):
    if dataset == 'cifar10':
        if data_transform is None:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ])
        dataset_path = os.path.join(dataset_path, 'cifar10')
        if split == 'train':
            train_data = datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=data_transform)
            train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
                                           drop_last=drop_last, num_workers=0, pin_memory=True)
            return train_data_loader
        elif split == 'std_test' or split == 'full_test':
            test_data = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=data_transform)
            test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0, pin_memory=True)
            return test_data_loader
        elif split == 'valid' or split == 'val':
            val_set_dir = os.path.join('clean_set', 'cifar10', 'clean_split')
            val_set_img_dir = os.path.join(val_set_dir, 'data')
            val_set_label_path = os.path.join(val_set_dir, 'clean_labels')
            val_set = IMG_Dataset(data_dir=val_set_img_dir, label_path=val_set_label_path, transforms=data_transform)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0, pin_memory=True)
            return val_loader
        elif split == 'test':
            test_set_dir = os.path.join('clean_set', 'cifar10', 'test_split')
            test_set_img_dir = os.path.join(test_set_dir, 'data')
            test_set_label_path = os.path.join(test_set_dir, 'labels')
            test_set = IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path, transforms=data_transform)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=0, pin_memory=True)
            return test_loader
    if dataset == 'cifar20':
        if data_transform is None:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ])
        dataset_path = os.path.join(dataset_path, 'cifar10')
        if split == 'train':
            train_data = CIFAR20(root=dataset_path, train=True, download=False, transform=data_transform)
            train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
                                           drop_last=drop_last, num_workers=0, pin_memory=True)
            return train_data_loader
        elif split == 'test':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ])
            test_data = CIFAR20(root=dataset_path, train=False, download=False, transform=data_transform)
            test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle,
                                           drop_last=drop_last, num_workers=0, pin_memory=True)
            return test_loader
    elif dataset == 'gtsrb':
        if data_transform is None:
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.3337, 0.3060, 0.3171], [0.2672, 0.2560, 0.2629]),
            ])
        dataset_path = os.path.join(dataset_path, 'gtsrb')
        if split == 'train':
            train_data = datasets.GTSRB(root=dataset_path, split='train', download=False, transform=data_transform)
            train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=4, pin_memory=True)
            return train_data_loader
        elif split == 'std_test' or split == 'full_test':
            test_data = datasets.GTSRB(root=dataset_path, split='test', download=False, transform=data_transform)
            test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0, pin_memory=True)
            return test_data_loader
        elif split == 'valid' or split == 'val':
            val_set_dir = os.path.join('clean_set', 'gtsrb', 'clean_split')
            val_set_img_dir = os.path.join(val_set_dir, 'data')
            val_set_label_path = os.path.join(val_set_dir, 'clean_labels')
            val_set = IMG_Dataset(data_dir=val_set_img_dir, label_path=val_set_label_path, transforms=data_transform)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0, pin_memory=True)
            return val_loader
        elif split == 'test':
            test_set_dir = os.path.join('clean_set', 'gtsrb', 'test_split')
            test_set_img_dir = os.path.join(test_set_dir, 'data')
            test_set_label_path = os.path.join(test_set_dir, 'labels')
            test_set = IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path, transforms=data_transform)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=0, pin_memory=True)
            return test_loader
    elif dataset == 'Imagenette':
        if data_transform is None:
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if split == 'train':
            train_data = datasets.ImageFolder(os.path.join(os.path.join(dataset_path, 'imagenette2-160'), 'train'), data_transform)
            train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0, pin_memory=True)
            return train_data_loader
        elif split == 'std_test' or split == 'full_test' or split == 'test':
            test_data = datasets.ImageFolder(os.path.join(os.path.join(dataset_path, 'imagenette2-160'), 'val'), data_transform)
            test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0, pin_memory=True)
            return test_data_loader
        elif split == 'valid' or split == 'val':
            val_set_dir = os.path.join('clean_set', 'imagenette', 'clean_split')
            val_set_img_dir = os.path.join(val_set_dir, 'data')
            val_set_label_path = os.path.join(val_set_dir, 'clean_labels')
            val_set = IMG_Dataset(data_dir=val_set_img_dir, label_path=val_set_label_path, transforms=data_transform)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0, pin_memory=True)
            return val_loader
        # elif split == 'test':
        #     test_set_dir = os.path.join('clean_set', 'imagenette', 'test_split')
        #     test_set_img_dir = os.path.join(test_set_dir, 'data')
        #     test_set_label_path = os.path.join(test_set_dir, 'labels')
        #     test_set = IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path, transforms=data_transform)
        #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=0, pin_memory=True)
        #     return test_loader
    elif dataset == 'speech_commands':
        if split == 'train':
            train_dataset = SpeechCommandsDataset(os.path.join('./data', 'speech_commands/speech_commands_v0.01'),
                                                  transform=data_transform)
            train_set = build_retain_sets_sc(train_dataset, num_classes=12)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                                  num_workers=0, pin_memory=True)
            return train_loader
        elif split =='valid':
            test_dataset = SpeechCommandsDataset(
                os.path.join('./data', 'speech_commands/speech_commands_test_set_v0.01'),
                transform=data_transform)
            test_set = build_retain_sets_sc(test_dataset, num_classes=12)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                                      drop_last=drop_last,
                                                      num_workers=0, pin_memory=True)
            return test_loader
        elif split == 'test':
            test_dataset = SpeechCommandsDataset(
                os.path.join('./data', 'speech_commands/speech_commands_test_set_v0.01'),
                transform=data_transform)
            test_set = build_retain_sets_sc(test_dataset, num_classes=12)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                              num_workers=0, pin_memory=True)
            return test_loader
    else:
        print('<To Be Implemented> Dataset = %s' % dataset)
        exit(0)

def generate_dataset(dataset='cifar10', dataset_path='./data/', batch_size=128, split='train',
                    data_transform=None):
    if dataset == 'cifar10':
        if data_transform is None:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ])
        dataset_path = os.path.join(dataset_path, 'cifar10')
        if split == 'train':
            train_data = datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=data_transform)
            return train_data
        elif split == 'std_test' or split == 'full_test':
            test_data = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=data_transform)
            return test_data
        elif split == 'valid' or split == 'val':
            val_set_dir = os.path.join('clean_set', 'cifar10', 'clean_split')
            val_set_img_dir = os.path.join(val_set_dir, 'data')
            val_set_label_path = os.path.join(val_set_dir, 'clean_labels')
            val_set = IMG_Dataset(data_dir=val_set_img_dir, label_path=val_set_label_path, transforms=data_transform)
            return val_set
        elif split == 'test':
            test_set_dir = os.path.join('clean_set', 'cifar10', 'test_split')
            test_set_img_dir = os.path.join(test_set_dir, 'data')
            test_set_label_path = os.path.join(test_set_dir, 'labels')
            test_set = IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path, transforms=data_transform)
            return test_set
    elif dataset == 'gtsrb':
        if data_transform is None:
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.3337, 0.3060, 0.3171], [0.2672, 0.2560, 0.2629]),
            ])
        dataset_path = os.path.join(dataset_path, 'gtsrb')
        if split == 'train':
            train_data = datasets.GTSRB(root=dataset_path, split='train', download=False, transform=data_transform)
            return train_data
        elif split == 'std_test' or split == 'full_test':
            test_data = datasets.GTSRB(root=dataset_path, split='test', download=False, transform=data_transform)
            return test_data
        elif split == 'valid' or split == 'val':
            val_set_dir = os.path.join('clean_set', 'gtsrb', 'clean_split')
            val_set_img_dir = os.path.join(val_set_dir, 'data')
            val_set_label_path = os.path.join(val_set_dir, 'clean_labels')
            val_set = IMG_Dataset(data_dir=val_set_img_dir, label_path=val_set_label_path, transforms=data_transform)
            return val_set
        elif split == 'test':
            test_set_dir = os.path.join('clean_set', 'gtsrb', 'test_split')
            test_set_img_dir = os.path.join(test_set_dir, 'data')
            test_set_label_path = os.path.join(test_set_dir, 'labels')
            test_set = IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path, transforms=data_transform)
            return test_set
    elif dataset == 'Imagenette':
        if data_transform is None:
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if split == 'train':
            train_data = datasets.ImageFolder(os.path.join(os.path.join(dataset_path, 'imagenette2-160'), 'train'), data_transform)
            return train_data
        elif split == 'std_test' or split == 'full_test' or split == 'test':
            test_data = datasets.ImageFolder(os.path.join(os.path.join(dataset_path, 'imagenette2-160'), 'val'), data_transform)
            return test_data
        elif split == 'valid' or split == 'val':
            val_set_dir = os.path.join('clean_set', 'imagenette', 'clean_split')
            val_set_img_dir = os.path.join(val_set_dir, 'data')
            val_set_label_path = os.path.join(val_set_dir, 'clean_labels')
            val_set = IMG_Dataset(data_dir=val_set_img_dir, label_path=val_set_label_path, transforms=data_transform)
            return val_set
    else:
        print('<To Be Implemented> Dataset = %s' % dataset)
        exit(0)

def unpack_poisoned_train_set(args, batch_size=128, shuffle=False, data_transform=None):
    """
    Return with `poison_set_dir`, `poisoned_set_loader`, `poison_indices`, and `cover_indices` if available
    """
    if args.dataset == 'cifar10':
        if data_transform is None:
            if args.no_normalize:
                data_transform = transforms.Compose([
                        transforms.ToTensor(),
                ])
            else:
                data_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
    elif args.dataset == 'gtsrb':
        if data_transform is None:
            if args.no_normalize:
                data_transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                ])
            else:
                data_transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]),
                ])
    elif args.dataset == 'imagenette':
        if data_transform is None:
            if args.no_normalize:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])    
    else: raise NotImplementedError()

    poison_set_dir = supervisor.get_poison_set_dir(args)

    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    cover_indices_path = os.path.join(poison_set_dir, 'cover_indices') # for adaptive attacks

    poisoned_set = IMG_Dataset(data_dir=poisoned_set_img_dir,
                                label_path=poisoned_set_label_path, transforms=data_transform)

    poisoned_set_loader = torch.utils.data.DataLoader(poisoned_set,
                                                      batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    poison_indices = torch.load(poison_indices_path)
    
    if ('adaptive' in args.poison_type) or args.poison_type == 'TaCT':
        cover_indices = torch.load(cover_indices_path)
        return poison_set_dir, poisoned_set_loader, poison_indices, cover_indices
    
    return poison_set_dir, poisoned_set_loader, poison_indices, []

def jaccard_idx(mask: torch.Tensor, real_mask: torch.Tensor, select_num: int = 9) -> float:
    if select_num <= 0: return 0
    mask = mask.to(dtype=torch.float)
    real_mask = real_mask.to(dtype=torch.float)
    detect_mask = mask > mask.flatten().topk(select_num)[0][-1]
    sum_temp = detect_mask.int() + real_mask.int()
    overlap = (sum_temp == 2).sum().float() / (sum_temp >= 1).sum().float()
    return float(overlap)

def normalize_mad(values: torch.Tensor, side: str = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / mad / 1.4826
    if side == 'double':    # TODO: use a loop to optimie code
        dev_list = []
        for i in range(len(values)):
            if values[i] <= median:
                dev_list.append(float(median - values[i]))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] <= median:
                measures[i] = abs_dev[i] / mad / 1.4826

        dev_list = []
        for i in range(len(values)):
            if values[i] >= median:
                dev_list.append(float(values[i] - median))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] >= median:
                measures[i] = abs_dev[i] / mad / 1.4826
    return measures

def to_list(x) -> list:
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x.tolist()
    return list(x)


def val_atk(args, model, split='test', batch_size=100):
    """
    Validate the attack (described in `args`) on `model`
    """
    print("------val_atk------")
    model.eval()
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'cifar20':
        if args.no_normalize:
            data_transform = transforms.Compose([
                    transforms.ToTensor(),
            ])
            trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
            ])
        else:
            data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
    elif args.dataset == 'Imagenette':
        if args.no_normalize:
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
        else:
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            trigger_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    elif args.dataset == 'speech_commands':
        n_mels = 32  # inputs of NN
        valid_feature_transform = transforms.Compose(
            [ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
        data_transform = transforms.Compose([LoadAudio(),
                                             FixAudioLength(),
                                             valid_feature_transform])
        trigger_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif args.dataset == 'gtsrb':
        if args.no_normalize:
            data_transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
            ])
            trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
            ])
        else:
            data_transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]),
            ])
            trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]),
            ])

    else: raise NotImplementedError()

    if ("ood" not in args.poison_type) and (args.poison_type != 'IB'):
        poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                           target_class=config.target_class[args.dataset],
                                                           trigger_transform=trigger_transform,
                                                           is_normalized_input=(not args.no_normalize),
                                                           alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                           trigger_name=args.trigger, args=args)
        test_loader = generate_dataloader(dataset=args.dataset, dataset_path=config.data_dir,
                                          batch_size=batch_size, split=split,
                                          shuffle=False, drop_last=False,
                                          data_transform=data_transform)

    if args.poison_type == 'none':
        num = 0
        num_non_target = 0
        num_clean_correct = 0

        acr = 0 # attack correct rate
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):

                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

        clean_acc = num_clean_correct / num
        print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
        
        return clean_acc, 0, clean_acc

    if args.poison_type == 'TaCT':
        num = 0
        num_source = 0
        num_non_source = 0
        num_clean_correct = 0
        num_poison_eq_clean_label = 0
        num_poison_eq_poison_label_source = 0
        num_poison_eq_poison_label_non_source = 0
        acr = 0 # attack correct rate
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):

                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

                # filter out target inputs (FIXME: target now fixed to 0)
                data = data[label != 0]
                label = label[label != 0]
                # source inputs (FIXME: source now fixed to 1)
                source_data = data[label == 1]
                source_label = label[label == 1]
                # non-source inputs
                non_source_data = data[label != 1]
                non_source_label = label[label != 1]

                num_source += len(source_label)
                num_non_source += len(non_source_label)

                # poison!
                if len(source_label) > 0: poison_source_data, poison_source_label = poison_transform.transform(source_data, source_label)
                if len(non_source_label) > 0: poison_non_source_data, poison_non_source_label = poison_transform.transform(non_source_data, non_source_label)

                # forward
                if len(source_label) > 0: poison_source_output = model(poison_source_data)
                if len(non_source_label) > 0: poison_non_source_output = model(poison_non_source_data)
                if len(source_label) > 0: poison_source_pred = poison_source_output.argmax(dim=1)  # get the index of the max log-probability
                if len(non_source_label) > 0: poison_non_source_pred = poison_non_source_output.argmax(dim=1)  # get the index of the max log-probability
                
                for bid in range(len(source_label)):
                    if poison_source_pred[bid] == poison_source_label[bid]:
                        num_poison_eq_poison_label_source+=1
                    if poison_source_pred[bid] == source_label[bid]:
                        num_poison_eq_clean_label+=1
                for bid in range(len(non_source_label)):
                    if poison_non_source_pred[bid] == poison_non_source_label[bid]:
                        num_poison_eq_poison_label_non_source+=1
                    if poison_non_source_pred[bid] == non_source_label[bid]:
                        num_poison_eq_clean_label+=1

        clean_acc = num_clean_correct / num
        asr_source = num_poison_eq_poison_label_source/num_source
        asr_non_source = num_poison_eq_poison_label_non_source/num_non_source
        acr = num_poison_eq_clean_label / len(test_loader.dataset)
        print('Accuracy : %d/%d = %f' % (num_clean_correct, num, clean_acc))
        print('ASR (source) : %d/%d = %f' % (num_poison_eq_poison_label_source, num_source, asr_source))
        print('ASR (non-source) : %d/%d = %f' % (num_poison_eq_poison_label_non_source, num_non_source, asr_non_source))
        print('ACR (Attack Correct Rate) : %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))

        return clean_acc, asr_source, asr_non_source, acr

    # elif args.poison_type == 'meadefender':
    #     CLASS_A = 0
    #     CLASS_B = 1
    #     CLASS_C = 2
    #     mixer = {
    #         "Half": HalfMixer(),
    #         "Vertical": RatioMixer(),
    #         "Diag": DiagnalMixer(),
    #         "RatioMix": RatioMixer(),
    #         "Donut": DonutMixer(),
    #         "Hot Dog": HotDogMixer(),
    #     }
    #     poi_set = generate_dataset(dataset=args.dataset, dataset_path=config.data_dir,
    #                                   batch_size=batch_size, split=split,
    #                                   data_transform=data_transform)
    #     poi_set = MixDataset(dataset=poi_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
    #                          data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1,
    #                          transform=None)  # constructed from test data
    #     poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=batch_size, shuffle=False)
    #     #test loader is used to test the accuracy
    #     num = 0
    #     num_non_target = 0
    #     num_clean_correct = 0
    #     num_poison_eq_poison_label = 0
    #     num_poison_eq_clean_label = 0
    #     with torch.no_grad():
    #         for batch_idx, (data, label) in enumerate(test_loader):
    #             data, label = data.cuda(), label.cuda()  # train set batch
    #             output = model(data)
    #             pred = output.argmax(dim=1)  # get the index of the max log-probability
    #             num_clean_correct += pred.eq(label).sum().item()
    #             num += len(label)
    #
    #         for _, (wm_data, wm_label, _) in enumerate(poi_loader):
    #             wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
    #             output = model(wm_data)
    #             pred = output.argmax(dim=1)  # get the index of the max log-probability
    #             num_poison_eq_poison_label += pred.eq(wm_label).sum().item()
    #             num_non_target += len(wm_label)
    #
    #     clean_acc = num_clean_correct / num
    #     asr = num_poison_eq_poison_label / num_non_target
    #     acr = num_poison_eq_clean_label / len(test_loader.dataset)
    #     print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
    #     print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
    #     print('ACR (Attack Correct Rate): %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))
    #     return clean_acc, asr, acr
    # elif 'mu_' in args.poison_type:
    #     forget_class = 4
    #     # test_set = generate_dataset(dataset=args.dataset, dataset_path=config.data_dir,
    #     #                            batch_size=batch_size, split=split,
    #     #                            data_transform=data_transform)#split = test
    #     test_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
    #                               download=True, transform=data_transform)
    #     classwise_test = get_classwise_ds(test_set, num_classes=10)
    #     clean_test_set = build_retain_sets_in_unlearning(classwise_test, 10, int(forget_class))
    #     poi_set = classwise_test[int(forget_class)]
    #
    #     poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=batch_size, shuffle=False)
    #     test_loader = torch.utils.data.DataLoader(dataset=clean_test_set, batch_size=batch_size, shuffle=False)
    #     num = 0
    #     num_non_target = 0
    #     num_clean_correct = 0
    #     num_poison_eq_poison_label = 0
    #     num_poison_eq_clean_label = 0
    #     with torch.no_grad():
    #         for batch_idx, (data, label) in enumerate(test_loader):
    #             data, label = data.cuda(), label.cuda()  # train set batch
    #             output = model(data)
    #             pred = output.argmax(dim=1)  # get the index of the max log-probability
    #             num_clean_correct += pred.eq(label).sum().item()
    #             num += len(label)
    #
    #         for _, (wm_data, wm_label) in enumerate(poi_loader):
    #             wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
    #             output = model(wm_data)
    #             pred = output.argmax(dim=1)  # get the index of the max log-probability
    #             num_poison_eq_poison_label += pred.eq(wm_label).sum().item()
    #             num_non_target += len(wm_label)
    #
    #     clean_acc = num_clean_correct / num
    #     asr = num_poison_eq_poison_label / num_non_target
    #     acr = num_poison_eq_clean_label / len(test_loader.dataset)
    #     print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
    #     print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
    #     print('ACR (Attack Correct Rate): %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))
    #     return clean_acc, asr, 0.0#, asr, acr

    # elif args.poison_type == 'entangled_watermark':
    #     if ewe_poison_loader == None:
    #         poison_set_dir = supervisor.get_poison_set_dir(args)
    #         poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    #         poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    #
    #         print('dataset : %s' % poisoned_set_img_dir)
    #
    #         poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
    #                                          label_path=poisoned_set_label_path,
    #                                          transforms=data_transform)
    #
    #         ewe_poison_loader = DataLoader(
    #             poisoned_set,
    #             batch_size=batch_size, shuffle=True)
    #     num = 0
    #     num_non_target = 0
    #     num_clean_correct = 0
    #     num_poison_eq_poison_label = 0
    #     num_poison_eq_clean_label = 0
    #     with torch.no_grad():
    #         for batch_idx, (data, label) in enumerate(test_loader):
    #             data, label = data.cuda(), label.cuda()  # train set batch
    #             output = model(data)
    #             pred = output.argmax(dim=1)  # get the index of the max log-probability
    #             num_clean_correct += pred.eq(label).sum().item()
    #             num += len(label)
    #
    #         for _, (wm_data, wm_label) in enumerate(ewe_poison_loader):
    #             wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
    #             output = model(wm_data)
    #             pred = output.argmax(dim=1)  # get the index of the max log-probability
    #             num_poison_eq_poison_label += pred.eq(wm_label).sum().item()
    #             num_non_target += len(wm_label)
    #
    #     clean_acc = num_clean_correct / num
    #     asr = num_poison_eq_poison_label / num_non_target
    #     acr = num_poison_eq_clean_label / len(test_loader.dataset)
    #     print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
    #     print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
    #     print('ACR (Attack Correct Rate): %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))
    #     return clean_acc, asr, acr
    elif (args.poison_type == 'entangled_watermark'
          or args.poison_type == 'UAE'):
        poison_set_dir = supervisor.get_poison_set_dir(args)
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
        poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

        print('dataset : %s' % poisoned_set_img_dir)

        poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                         label_path=poisoned_set_label_path,
                                         transforms=data_transform)
        poison_loader = DataLoader(
            poisoned_set,
            batch_size=batch_size, shuffle=False)

        num = 0
        num_non_target = 0
        num_clean_correct = 0
        num_poison_eq_poison_label = 0
        num_poison_eq_clean_label = 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

            for _, (wm_data, wm_label) in enumerate(poison_loader):
                wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
                output = model(wm_data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_poison_eq_poison_label += pred.eq(wm_label).sum().item()
                num_non_target += len(wm_label)

        clean_acc = num_clean_correct / num
        asr = num_poison_eq_poison_label / num_non_target
        acr = num_poison_eq_clean_label / len(test_loader.dataset)
        print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
        print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
        print('ACR (Attack Correct Rate): %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))
        return clean_acc, asr, acr

    elif args.poison_type == 'meadefender':

        CLASS_A = 0  # TODO 0
        CLASS_B = 1 # TODO 1
        CLASS_C = 2
        # mixer = {
        #     "Half": HalfMixer()
        # }
        # poisoned_set = generate_dataset(dataset=args.dataset, dataset_path=config.data_dir,
        #                            batch_size=batch_size, split='test',
        #                            data_transform=data_transform)

        # poisoned_set = MixDataset(dataset=poisoned_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
        #                      data_rate=0.2, normal_rate=0, mix_rate=0, poison_rate=1.0,
        #                      transform=None)

        if args.dataset == 'cifar10':
            test_set = datasets.CIFAR10(os.path.join(config.data_dir, 'cifar10'), train=False,
                                        download=True, transform=data_transform)
        elif args.dataset == 'Imagenette':
            test_set = datasets.ImageFolder('./data/imagenette2-160/val', transform=data_transform)

        mixer = HalfMixer()
        poisoned_set = com_MixDataset(dataset=test_set, mixer=mixer,
                                    classA=CLASS_A, classB=CLASS_B,
                                    classC=CLASS_C, data_rate=1,
                                    normal_rate=0, mix_rate=0,
                                    poison_rate=1)
        poison_loader = DataLoader(
            poisoned_set,
            batch_size=batch_size, shuffle=False)

        num = 0
        num_non_target = 0
        num_clean_correct = 0
        num_poison_eq_poison_label = 0
        num_poison_eq_clean_label = 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

            for _, (wm_data, wm_label) in enumerate(poison_loader):
                wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
                output = model(wm_data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_poison_eq_poison_label += pred.eq(wm_label).sum().item()
                num_non_target += len(wm_label)

        clean_acc = num_clean_correct / num
        asr = num_poison_eq_poison_label / num_non_target
        acr = num_poison_eq_clean_label / len(test_loader.dataset)
        print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
        print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
        print('ACR (Attack Correct Rate): %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))
        return clean_acc, asr, acr

    elif args.poison_type == 'MBW':
        watermark = queries.StochasticImageQuery(query_size=(10, 3, 32, 32),
                                                 response_size=(10,), query_scale=255, response_scale=10,)
        # watermark.query = torch.load(os.path.join(supervisor.get_poison_set_dir(args), 'watermark_query.pt'))
        watermark = torch.load(os.path.join(supervisor.get_poison_set_dir(args), 'watermark.pt'))
        watermark.eval()
        wm_samples, wm_response = watermark(discretize=True)

        num = 0
        num_non_target = 0
        num_clean_correct = 0
        num_poison_eq_poison_label = 0
        num_poison_eq_clean_label = 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

            wm_data, wm_label = wm_samples.cuda(), wm_response.cuda()  # train set batch
            output_wm = model(wm_data)
            pred_wm = output_wm.argmax(dim=1)  # get the index of the max log-probability
            num_poison_eq_poison_label += pred_wm.eq(wm_label).sum().item()
            num_non_target += len(wm_label)

        clean_acc = num_clean_correct / num
        asr = num_poison_eq_poison_label / num_non_target
        acr = num_poison_eq_clean_label / len(test_loader.dataset)
        print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
        print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
        print('ACR (Attack Correct Rate): %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))
        return clean_acc, asr, acr

    elif 'ood' in args.poison_type:
        if args.dataset == 'Imagenet_21':
            train_set_imagenet = Imagenet64(root=os.path.join('./data', 'Imagenet64_subset-21'),
                                            transforms=data_transform, train=True)
            test_set_imagenet = Imagenet64(root=os.path.join('./data', 'Imagenet64_subset-21'), transforms=data_transform,
                                           train=False)
            # train_set = build_retain_sets_in_unlearning(train_set_imagenet, num_classes=20 + 1,
            #                                             forget_class=20)
            # test_set = build_retain_sets_in_unlearning(test_set_imagenet, num_classes=20 + 1, forget_class=20)

            ood_train_set = get_classwise_ds(train_set_imagenet, 20 + 1)[20]
            ood_train_set = OodDataset(ood_train_set, label=20)

        elif args.dataset == 'cifar10':
            test_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
                                        download=True, transform=data_transform)
            # ood_train_set = CIFAR100(os.path.join('./data', 'cifar100'), train=True,
            #                          download=True, transform=data_transform)
            # In CIFAR-100, 'flowers' includes 54, 62, 70, 82, 92
            # flower_indices_train = [i for i, label in enumerate(ood_train_set.targets) if label in [54, 62, 70, 82, 92]]
            # ood_train_set = OodDataset(ood_train_set, flower_indices_train, label=20)
            # download the dataset
            poison_set_dir = supervisor.get_poison_set_dir(args)
            poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
            poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

            ood_train_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                              label_path=poisoned_set_label_path,
                                              transforms=data_transform)
            #for sub ood class
            if 'sub' in args.poison_type:
                target_label = 0
            else:
                target_label = 10
            ood_train_set = OodDataset_from_Dataset(dataset=ood_train_set, label=target_label)

        elif args.dataset == 'cifar20':
            test_set = CIFAR20(os.path.join('./data', 'cifar100'), train=False,
                                        download=True, transform=data_transform)
            # download the dataset
            poison_set_dir = supervisor.get_poison_set_dir(args)
            poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
            poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

            ood_train_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                              label_path=poisoned_set_label_path,
                                              transforms=data_transform)
        elif args.dataset == 'speech_commands':
            test_dataset = SpeechCommandsDataset(
                os.path.join('./data', 'speech_commands/speech_commands_test_set_v0.01'),
                transform=data_transform)
            test_set = build_retain_sets_sc(test_dataset, num_classes=12)

            ood_train_dataset = SpeechCommandsDataset(
                os.path.join('./data', 'speech_commands/speech_commands_v0.01_ood'),
                transform=data_transform)
            ood_train_set = build_ood_sets_sc(ood_train_dataset, 12)
        elif args.dataset == 'Imagenette':
            test_set = datasets.ImageFolder('./data/imagenette2-160/val', transform=data_transform)
            plain_ood_dataset = FixedLabelDataset(
                root_dir='./data/filtered_200',
                label=0, #TODO
                transform=data_transform)
            #TODO subset_size: 150
            subset_size = 150
            selected_indices = torch.load(f'imagenette_{subset_size}_indices.pt')
            ood_train_set = Subset(plain_ood_dataset, selected_indices)

        poi_loader = torch.utils.data.DataLoader(dataset=ood_train_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
        num = 0
        num_non_target = 0
        num_clean_correct = 0
        num_poison_eq_poison_label = 0
        num_poison_eq_clean_label = 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

            for _, (wm_data, wm_label) in enumerate(poi_loader):
                wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
                output = model(wm_data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_poison_eq_poison_label += pred.eq(wm_label).sum().item()
                num_non_target += len(wm_label)

        clean_acc = num_clean_correct / num
        asr = num_poison_eq_poison_label / num_non_target
        acr = num_poison_eq_clean_label / len(test_loader.dataset)
        print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
        print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
        print('ACR (Attack Correct Rate): %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))
        return clean_acc, asr, 0.0  # , asr, acr

    elif args.poison_type == 'IB':
        test_set = datasets.GTSRB(os.path.join('./data', 'gtsrb'), split='test',
                                  transform=data_transform, download=True)
        img_size = 128

        test_loader = torch.utils.data.DataLoader(
            # test_set,
            Subset(test_set, list(range(int(len(test_set) / 3)))),
            batch_size=32, shuffle=False)

        Target_labels = torch.stack([i * torch.ones(1) for i in range(1)]).expand(
            1, batch_size).permute(1, 0).reshape(-1, 1).squeeze().to(dtype=torch.long, device='cuda')
        model.eval()
        TriggerNet= U_Net(output_ch=6)
        TriggerNet.load_state_dict(torch.load(os.path.join(supervisor.get_poison_set_dir(), 'full_base_aug_seed=2333_Tri.pt')))
        TriggerNet.eval()

        EmbbedNet = Embbed()
        EmbbedNet = EmbbedNet.cuda()

        Correct = 0
        Tot = 0
        for fs, labels in test_loader:
            fs = fs.to(dtype=torch.float).cuda()
            labels = labels.to(dtype=torch.long).cuda(
            ).view(-1, 1).squeeze().squeeze()
            out, _ = model(fs, feature=True)
            _, predicts = out.max(1)
            Correct += predicts.eq(labels).sum().item()
            Tot += fs.shape[0]

        print('Eval-normal: Acc:{:.2f}'.format( 100 * Correct / Tot))

        Correct = 0
        Tot = 0

        for fs, labels in test_loader:
            fs = fs.to(dtype=torch.float).cuda()
            Triggers = TriggerNet(fs)
            Triggers = EmbbedNet(Triggers[:, 0:3 * 1, :, :],
                                 Triggers[:, 3 * 1:6 * 1, :, :])
            Triggers = torch.round(Triggers) / 255
            fs = fs.unsqueeze(1).expand(fs.shape[0], 1, 3, img_size,
                                        img_size).reshape(-1, 3, img_size, img_size)
            Triggers = Triggers.reshape(-1, 3, img_size, img_size)
            fs = fs + Triggers
            fs = torch.clip(fs, min=0, max=1)
            out, _ = model(fs, feature=True)
            _, predicts = out.max(1)
            Correct += predicts.eq(Target_labels[:len(out)]).sum().item()
            Tot += fs.shape[0]
        Acc = 100 * Correct / Tot

        print(
            'Eval-poison Acc:{:.2f}'.format(Acc))

    elif args.poison_type == 'composite_backdoor':
        if args.dataset == 'cifar10': #TODO
            test_set = datasets.CIFAR10(os.path.join(config.data_dir, 'cifar10'), train=False,
                                        download=True, transform=data_transform)
        elif args.dataset == 'Imagenette':
            test_set = datasets.ImageFolder('./data/imagenette2-160/val', transform=data_transform)

        CLASS_A = 0
        CLASS_B = 1
        CLASS_C = 2  # A + B -> C

        mixer = HalfMixer()
        poison_set = com_MixDataset(dataset=test_set, mixer=mixer,
                                classA=CLASS_A, classB=CLASS_B,
                                classC=CLASS_C, data_rate=1,
                                normal_rate=0, mix_rate=0,
                                poison_rate=1)

        poison_loader = DataLoader(dataset=poison_set, num_workers=0,
                                   batch_size=batch_size)
        test_loader = DataLoader(dataset=test_set, num_workers=0,
                                 batch_size=batch_size)

        with torch.no_grad():
            num = 0
            num_non_target = 0
            num_clean_correct = 0
            num_poison_eq_poison_label=0
            num_poison_eq_clean_label = 0
            for _, (x_batch, y_batch) in enumerate(test_loader):
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                output = model(x_batch)
                pred = output.max(dim=1)[1]

                num += x_batch.size(0)
                num_clean_correct += (pred == y_batch).sum().item()

            for _, (x_batch, y_batch) in enumerate(poison_loader):
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                output = model(x_batch)
                pred = output.max(dim=1)[1]

                num_poison_eq_poison_label += pred.eq(y_batch).sum().item()
                num_non_target += len(y_batch)

            clean_acc = num_clean_correct / num
            asr = num_poison_eq_poison_label / num_non_target
            acr = num_poison_eq_clean_label / len(test_loader.dataset)
            print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
            print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
            return clean_acc, asr, acr
    else:
        num = 0
        num_non_target = 0
        num_clean_correct = 0
        num_poison_eq_poison_label = 0
        num_poison_eq_clean_label = 0

        acr = 0 # attack correct rate
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):

                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

                data, poison_label = poison_transform.transform(data, label)
                poison_output = model(data)
                poison_pred = poison_output.argmax(dim=1)  # get the index of the max log-probability
                this_batch_size = len(poison_label)
                for bid in range(this_batch_size):
                    if label[bid] != poison_label[bid]: # samples of non-target classes
                        num_non_target += 1
                        if poison_pred[bid] == poison_label[bid]:
                            num_poison_eq_poison_label+=1
                        if poison_pred[bid] == label[bid]:
                            num_poison_eq_clean_label+=1
                    else:
                        if poison_pred[bid] == label[bid]:
                            num_poison_eq_clean_label+=1

        clean_acc = num_clean_correct / num
        asr = num_poison_eq_poison_label / num_non_target
        acr = num_poison_eq_clean_label / len(test_loader.dataset)
        print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc))
        print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr))
        print('ACR (Attack Correct Rate): %d/%d = %f' % (num_poison_eq_clean_label, len(test_loader.dataset), acr))
        return clean_acc, asr, acr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w))

        for n in range(self.n_holes):
            y = torch.randint(high=h, size=(1, 1))
            x = torch.randint(high=w, size=(1, 1))

            y1 = torch.clip(y - self.length // 2, 0, h)
            y2 = torch.clip(y + self.length // 2, 0, h)
            x1 = torch.clip(x - self.length // 2, 0, w)
            x2 = torch.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img
