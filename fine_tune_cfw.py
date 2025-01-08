'''
codes used to class-feature watermark(cfw) fine-tuning phase
'''
import argparse
import copy
import logging
import os, sys
import time
from collections import OrderedDict

from datasets import load_dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from transformers import BertTokenizer

import cfw_finetuning
import dpcnn
import mobilenetv2
import resnet
import vgg
from text_utils.ag_newslike import Dbpedia, AG_NEWS
from cifar20 import CIFAR20, OODDataset, CombinedDataset, Dataset4List, CombinedDatasetText, \
    OodDataset_from_Dataset
from data_prep import DataTensorLoader, get_data_dbpedia
from imagenet import Imagenet64
from mu_utils import get_classwise_ds, get_one_classwise_ds, build_retain_sets_in_unlearning
from text_utils.text_processor import AGNewsProcessor, Dbpedia_Processor
from tools import val_atk
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='sub_ood_class')
parser.add_argument('-poison_rate', type=float,  required=False,
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-ember_options', type=str, required=False,
                    choices=['constrained', 'unconstrained', 'none'],
                    default='unconstrained')
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-resume', type=bool, default=False)
# parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
#model name
parser.add_argument('-model', type=str, default='model_unlearned.pt')
parser.add_argument('-unlearned_method', type=str, default='blindspot')
parser.add_argument('-un_model_sufix', type=str, default='_tr1')

parser.add_argument('-tri', type=str, default='0')
parser.add_argument('-target_class', type=str, default='0')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools
import time
import numpy as np

if args.dataset == 'cifar10':

    data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

elif args.dataset == 'Imagenet_21':

    data_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

elif args.dataset == 'cifar100' or args.dataset == 'cifar20':
    CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

elif args.dataset == 'gtsrb':

    data_transform_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

elif args.dataset == 'imagenet':
    print('[ImageNet]')

elif args.dataset == 'ember':
    print('[Non-image Dataset] Ember')

elif args.dataset == 'ag_news':
    print("ag_news aug")

else:
    raise NotImplementedError('dataset %s not supported' % args.dataset)


if args.dataset == 'cifar10':

    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'Imagenet_21':

    num_classes = 20
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'cifar20':

    num_classes = 20
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'cifar100':

    num_classes = 100
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'gtsrb':

    num_classes = 43
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([30, 60])
    learning_rate = 0.01
    batch_size = 128

elif args.dataset == 'imagenet':

    num_classes = 1000
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 90
    milestones = torch.tensor([30, 60])
    learning_rate = 0.1
    batch_size = 256

elif args.dataset == 'ember':

    num_classes = 2
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-6
    epochs = 10
    learning_rate = 0.1
    milestones = torch.tensor([])
    batch_size = 512

elif args.dataset == 'ag_news':
    num_classes = 4
    arch = config.arch[args.dataset]
    weight_decay = 1e-4#1e-8
    epochs = 3
    learning_rate = 1e-2#0.000001
    milestones = torch.tensor([10])
    batch_size = 16

else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)

milestones = milestones.tolist()

if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 0, 'pin_memory': True}

def valid_model(model, epoch):
    model.eval()
    # correct_by_class = {i: 0 for i in range(num_classes+1)}
    # total_by_class = {i: 0 for i in range(num_classes+1)}
    num = 0
    num_non_target = 0
    num_clean_correct = 0
    num_poison_eq_poison_label = 0
    num_poison_eq_poison_label_train = 0
    num_non_target_train = 0
    with torch.no_grad():
        if args.dataset == 'ag_news' and arch != dpcnn.DPCNN:
            for batch_idx, (id, mask, label) in enumerate(test_loader):
                id = id.cuda()
                mask = mask.cuda()
                label = label.cuda()
                output = model(id, mask)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)
            for wm_id, wm_mask, wm_label in poi_train_loader:
                wm_id, wm_mask, wm_label = wm_id.cuda(), wm_mask.cuda(), wm_label.cuda()  # train set batch
                output = model(wm_id, wm_mask)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_poison_eq_poison_label_train += pred.eq(wm_label).sum().item()
                num_non_target_train += len(wm_label)
        else:
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

                # for i in range(num_classes):
                #     correct_by_class[i] += ((pred == i) & (label == i)).sum().item()
                #     total_by_class[i] += (label == i).sum().item()

            for wm_data, wm_label in poi_train_loader:
                wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
                output = model(wm_data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_poison_eq_poison_label_train += pred.eq(wm_label).sum().item()
                num_non_target_train += len(wm_label)

        # for _, (wm_data, wm_label) in enumerate(poi_loader):
        #     # print("wm_label,", wm_label)
        #     wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
        #     output_wm = model(wm_data)
        #     pred_wm = output_wm.argmax(dim=1)  # get the index of the max log-probability
        #     num_poison_eq_poison_label += pred_wm.eq(wm_label).sum().item()
        #     num_non_target += len(wm_label)

    clean_acc = num_clean_correct / num
    # asr = num_poison_eq_poison_label / num_non_target
    train_asr = num_poison_eq_poison_label_train / num_non_target_train
    print("[epoch] ", epoch)
    print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc * 100), '%')

    # for i in range(num_classes+1):
    #     if total_by_class[i] > 0:
    #         accuracy = correct_by_class[i] / total_by_class[i]
    #         print(f"Class {i} Accuracy {correct_by_class[i]}/{total_by_class[i]}: {accuracy * 100:.4f}%")
    #     else:
    #         print(f"Class {i} Accuracy: N/A (no samples)")

    print('Training ASR: %d/%d = %f' % (num_poison_eq_poison_label_train, num_non_target_train, train_asr * 100), '%')
    # print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr*100), '%')
    return clean_acc * 100, train_asr * 100

if __name__ == '__main__':
    model_path = supervisor.get_poison_set_dir(args)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(model_path, f'output_{args.dataset}_unlearned.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    if args.dataset == 'cifar10':
        train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                download=True, transform=data_transform_aug)
        test_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
                                download=True, transform=data_transform)

        # download the dataset
        poison_set_dir = supervisor.get_poison_set_dir(args)
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
        poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

        ood_train_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                          label_path=poisoned_set_label_path,
                                          transforms=data_transform)

        # ood_train_set = Subset(ood_train_set, list(range(100)))
        # TODO change the label of ood_train_set
        target_label = int(args.target_class)#0
        ood_train_set = OodDataset_from_Dataset(dataset=ood_train_set, label=target_label)

        wm_set = CombinedDataset(train_set, ood_train_set)

        # ood_train_set = ImageNet(root=osp.join('./data', 'Imagenet64_subset-1'), train=True, original_transforms=data_transform_aug)
        # ood_test_set = ImageNet(root=osp.join('./data', 'Imagenet64_subset-1'), train=False, original_transforms=data_transform)


    train_loader = DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs
    )

    poi_loader = DataLoader(
        ood_train_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs
    )

    poi_train_loader = DataLoader(
        ood_train_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


    # Set Up Test Set for Debug & Evaluation
    # Train Code
    if 'resnet18' in args.model:
        arch = resnet.resnet18
    elif 'vgg19_bn' in args.model:
        arch = vgg.vgg19_bn
    elif 'mobilenetv2' in args.model:
        arch = mobilenetv2.mobilenetv2

    model = arch(num_classes=num_classes)
    unlearning_teacher = copy.deepcopy(model)

    state_dict = torch.load(supervisor.get_model_dir(args))
    model.load_state_dict(state_dict)
    model = model.cuda()

    unlearning_teacher = unlearning_teacher.cuda()

    # evaluate
    clean_acc, poison_acc = valid_model(model, epoch=0)
    logger.info(f'Target Model: \t Clean Acc \t {clean_acc} \t OOD Acc \t {poison_acc}')

    #TODO
    args.model = f'model_unlearned_sub_{int(num_classes)}_tri{args.tri}_cls{args.target_class}{args.un_model_sufix}.pt'
    # args.model = f'model_unlearned_sub_{num_classes}{args.un_model_sufix}.pt'

    model_path = supervisor.get_model_dir(args)
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")

    logger.info(f'Unlearned method: \t {args.unlearned_method}')
    model_size_scaler = 1

    kwargs = {
        "model": model,
        "unlearning_teacher": unlearning_teacher,
        "retain_train_dl": train_loader,
        "retain_valid_dl": test_loader,
        "forget_train_dl": poi_train_loader,
        "forget_valid_dl": poi_loader,
        "dampening_constant": 1,
        "selection_weighting": 10 * model_size_scaler,
        "forget_class": int(num_classes),
        "num_classes": num_classes,
        "dataset_name": args.dataset,
        "device": "cuda",
        "weights_path": model_path,
        "model_name": config.arch[args.dataset],
    }

    print("unlearn_ood_sub_class")
    # executes the method passed via args
    (d_t, d_f, d_r, mia), time_elapsed = getattr(cfw_finetuning, args.ft_method)(**kwargs)

    logger.info("Test Accuracy \t OOD Accuracy \t Train Accuracy")
    logger.info(f"{d_t} \t {d_f} \t {d_r}")

