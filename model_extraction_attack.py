'''
codes used to perform model extraction
'''
import argparse
import copy
import logging
import os, sys
from collections import OrderedDict

from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import CIFAR100
from tqdm import tqdm

import forget_full_class_strategies
import mobilenetv2
import resnet
import vgg
import wideResNet
import wresnet
from cifar20 import CIFAR20, OODDataset, CombinedDataset, OodDataset
from imagenet import Imagenet64
from mea.dfme import dfme
from mea.jacobian import jacobian_dataset_augmentation
from mu_utils import get_classwise_ds, get_one_classwise_ds, build_retain_sets_in_unlearning, build_retain_sets_sc, \
    build_ood_sets_sc
from speech_commands import CLASSES, SpeechCommandsDataset
from speech_transforms import ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, ToSTFT, ToTensor, \
    ToMelSpectrogram, LoadAudio, DeleteSTFT, ToMelSpectrogramFromSTFT, FixSTFTDimension, StretchAudioOnSTFT, \
    TimeshiftAudioOnSTFT
from tools import val_atk

from unlearn import evaluate
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])

parser.add_argument('-poison_type', type=str, required=False)#,
                    # default=default_args.parser_default['poison_type'],
                    # choices=default_args.parser_choices['poison_type'])

parser.add_argument('-poison_rate', type=float,  required=False)#,
                    # choices=default_args.parser_choices['poison_rate'],
                    # default=default_args.parser_default['poison_rate'])

parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
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
parser.add_argument('-model', type=str, default=None)
parser.add_argument('-mea_type', type=str, default='pb', choices=['pb','syn', 'df'])
parser.add_argument('-copy_arch', type=str, default=None)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn, optim
import torch
from utils import supervisor, tools
import numpy as np
import time
import torch.nn.functional as F

if args.trigger is None:
    if args.dataset != 'imagenet':
        args.trigger = config.trigger_default.get(args.poison_type, 'none')
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers.get(args.poison_type, 'none')

all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True

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

elif args.dataset == 'speech_commands':
    n_mels = 32 #inputs of NN

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

    valid_feature_transform = transforms.Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    data_transform = transforms.Compose([LoadAudio(),
                                         FixAudioLength(),
                                         valid_feature_transform])

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



else:
    raise NotImplementedError('dataset %s not supported' % args.dataset)


if args.dataset == 'cifar10':

    num_classes = 10
    if args.copy_arch is not None:
        if 'resnet18' in args.copy_arch:
            arch = resnet.resnet18
        elif 'vgg19_bn' in args.copy_arch:
            arch = vgg.vgg19_bn
        elif 'mobilenetv2' in args.copy_arch:
            arch = mobilenetv2.mobilenetv2
    else:
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

elif args.dataset == 'speech_commands':
    num_classes = len(CLASSES)
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-2
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 1e-3
    batch_size = 128

else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)

if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 0, 'pin_memory': True}

# test_loader
if args.dataset == 'Imagenet_21':
    test_set_imagenet = Imagenet64(root=osp.join('./data', 'Imagenet64_subset-21'), transforms=data_transform,
                                    train=False)
    test_set = build_retain_sets_in_unlearning(test_set_imagenet, num_classes=num_classes+1, forget_class=20)

elif args.dataset == 'cifar10':
    test_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
                                    download=True, transform=data_transform)
elif args.dataset == 'cifar100':
    test_set = datasets.CIFAR100(os.path.join('./data', 'cifar100'), train=False,
                                download=True, transform=data_transform)
elif args.dataset == 'cifar20':
    test_set = CIFAR20(os.path.join('./data', 'cifar100'), train=False,
                                download=True, transform=data_transform)

elif args.dataset == 'speech_commands':
    test_dataset = SpeechCommandsDataset(os.path.join('./data', 'speech_commands/speech_commands_test_set_v0.01'),
                                          transform=data_transform)

    test_set = build_retain_sets_sc(test_dataset, num_classes=num_classes)

    # print('test_set[0]', test_set[0][0].shape)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs
)

milestones = milestones.tolist()

def get_evaluation(
    model,
    wm_model,
    test_loader,
    device,
):
    model.eval()
    wm_model.eval()

    retain_acc_dict = evaluate(model, test_loader, device)
    predictions1 = []
    predictions2 = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            outputs1 = wm_model(inputs)
            outputs2 = model(inputs)
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)

            predictions1.append(predicted1.cpu())
            predictions2.append(predicted2.cpu())
        predictions1 = torch.cat(predictions1).flatten()
        predictions2 = torch.cat(predictions2).flatten()

        fidelity = accuracy_score(predictions1.numpy(), predictions2.numpy()) * 100.

    return retain_acc_dict["Acc"], fidelity

def mea_train(adv_loader):
    start_time = time.perf_counter()
    copy_model.train()
    train_loss = 0.0
    it = 0
    for data, label in tqdm(adv_loader):  # retain_train_dl
        data = data.cuda()
        # target = label.cuda()
        target = torch.softmax(victim_model(data), dim=1)  # .argmax(dim=1) #why? only use target1
        # target = torch.argmax(victim_model(data), dim=1)
        optimizer.zero_grad()

        # with autocast():
        output = copy_model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        it += 1

        loss.backward()
        optimizer.step()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(),
                                                                                                 optimizer.param_groups[
                                                                                                     0]['lr'],
                                                                                                 elapsed_time))

    return elapsed_time, train_loss/it

def valid_model(model, epoch):
    criterion = nn.CrossEntropyLoss().cuda()
    model.eval()
    # correct_by_class = {i: 0 for i in range(num_classes+1)}
    # total_by_class = {i: 0 for i in range(num_classes+1)}
    num = 0
    num_clean_correct = 0
    running_loss = 0.0
    it = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.cuda(), label.cuda()  # train set batch
            output = model(data)
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            num_clean_correct += pred.eq(label).sum().item()
            num += len(label)
            loss = criterion(output, label)
            it +=1

            # for i in range(num_classes):
            #     correct_by_class[i] += ((pred == i) & (label == i)).sum().item()
            #     total_by_class[i] += (label == i).sum().item()
            running_loss += loss.item()

    clean_acc = num_clean_correct / num

    print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc*100), '%')

    epoch_loss = running_loss/it
    return clean_acc*100, epoch_loss

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
                os.path.join(model_path, f'output_{args.mea_type}_extract_{supervisor.get_model_name(args)}.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    # victim model
    # args.model = f'model_w_ood_{num_classes}.pt'
    # if args.poison_type == 'ood_class':
    #     args.model = f'model_unlearned_{num_classes}.pt'
    if 'resnet18' in args.model:
        victim_arch = resnet.resnet18
    elif 'vgg19_bn' in args.model:
        victim_arch = vgg.vgg19_bn
    elif 'mobilenetv2' in args.model:
        victim_arch = mobilenetv2.mobilenetv2
    elif 'wrn_28_10' in args.model:
        victim_arch = wideResNet.wrn_28_10
    else:
        victim_arch = resnet.resnet18#arch
    victim_model_path = supervisor.get_model_dir(args)

    if args.dataset == 'speech_commands':
        victim_model = victim_arch(num_classes=num_classes, in_channels=1)
        copy_model = arch(num_classes=num_classes, in_channels=1)
    else:
        if '_11' in args.model:
            victim_model = victim_arch(num_classes=num_classes+1)
            copy_model = arch(num_classes=num_classes+1)
        else:
            victim_model = victim_arch(num_classes=num_classes)
            copy_model = arch(num_classes=num_classes)

    victim_model = nn.DataParallel(victim_model)
    copy_model = nn.DataParallel(copy_model)

    state_dict = torch.load(victim_model_path)
    try:
        victim_model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict['module.' + k] = v
        victim_model.load_state_dict(new_state_dict)

    victim_model = victim_model.cuda()
    victim_model.eval()
    copy_model = copy_model.cuda()
    # copy_model = copy.deepcopy(victim_model)

    if args.poison_type == 'ood_class' or args.poison_type == 'sub_ood_class':
        if args.copy_arch is not None:
            args.model = f'extract_{args.mea_type}_' + args.copy_arch + '_' + args.model
        else:
            args.model = f'extract_{args.mea_type}_' + args.model
    else:
        args.model = f'extract_{args.mea_type}_' + supervisor.get_model_name(args)
    output_path = supervisor.get_model_dir(args)
    print("model will save to: ", output_path)

    if args.mea_type == 'pb':
        if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'cifar20':
            adversarial_dataset = Imagenet64(root=osp.join('./data', 'Imagenet64_subset-21'),
                                             transforms=data_transform_aug,
                                             train=True)
            train_dataset = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                    download=True, transform=data_transform)

        elif args.dataset == 'speech_commands':
            train_dataset = SpeechCommandsDataset(os.path.join('./data', 'speech_commands/speech_commands_v0.02'),
                                                  transform=data_transform_aug)
            adversarial_dataset = build_retain_sets_sc(train_dataset, num_classes=num_classes)
            print("adversarial_dataset", len(adversarial_dataset))

        if os.path.exists(supervisor.get_poison_set_dir(args)+'/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(args)+'/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(train_dataset), 2500, replace=False) #5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(args)+'/domain_set_indices.np', training_indices)

        domain_dataset = Subset(train_dataset, training_indices)

        adversarial_loader = DataLoader(
            CombinedDataset(adversarial_dataset, domain_dataset),
            batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs
        )#adversarial loader


        optimizer = torch.optim.SGD(copy_model.parameters(), learning_rate, momentum=momentum,
                                    weight_decay=weight_decay)
        if args.dataset == 'speech_commands':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                                   factor=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

        criterion = nn.CrossEntropyLoss().cuda()
        logger.info('-----------Extract Model--------------')
        logger.info('Epoch \t lr \t Time \t TrainLoss \t ACC \t Fidelity')

        best_fidelity = 0.0
        for epoch in range(1, 100 + 1):  #TODO epochs # train backdoored base model
            # scheduler.step(epoch)

            elapsed_time, loss = mea_train(adversarial_loader)

            with torch.no_grad():
                test_acc, fidelity = get_evaluation(
                    copy_model,
                    victim_model,
                    test_loader,
                    'cuda',
                )
                val_atk(args, copy_model)

                logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                    epoch, optimizer.param_groups[0]['lr'], elapsed_time, loss, test_acc,
                    fidelity))

            if fidelity > best_fidelity:
                torch.save(copy_model.module.state_dict(), output_path)
                best_fidelity = fidelity

            if args.dataset == 'speech_commands':
                scheduler.step(metrics=loss)
            else:
                scheduler.step()

    elif args.mea_type == 'syn':
        # adv_dataset = Imagenet64(root=osp.join('./data', 'Imagenet64_subset-21'), transforms=data_transform_aug,
        #                                  train=True)
        if args.dataset == 'cifar10':
            adv_dataset = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                    download=True, transform=data_transform)
        elif args.dataset == 'speech_commands':
            train_dataset = SpeechCommandsDataset(os.path.join('./data', 'speech_commands/speech_commands_v0.01'),
                                                 transform=data_transform_aug)
            adv_dataset = build_retain_sets_sc(train_dataset, num_classes=num_classes)

        adv_dataset = Subset(adv_dataset, np.random.randint(0, len(adv_dataset), size=(len(adv_dataset)/4)))

        optimizer = torch.optim.SGD(copy_model.parameters(), learning_rate, momentum=momentum,
                                    weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss().cuda()
        logger.info('-----------Extract Model--------------')
        logger.info('Epoch \t lr \t Time \t TrainLoss \t ACC \t Fidelity')

        p_epochs = 6  # Number of substitute epochs
        epochs = 50  # Number of epochs to train the model at each substitute epoch
        lambda_ = 0.1  # Parameter for Jacobian-based dataset augmentation

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        best_fidelity = 0.0
        for p in range(p_epochs + 1):
            adv_loader = DataLoader(
                adv_dataset,
                batch_size=batch_size, shuffle=False,
                worker_init_fn=tools.worker_init, **kwargs
            )
            for epoch in range(1, epochs + 1):  # train backdoored base model
                elapsed_time, loss = mea_train(adv_loader)

                with torch.no_grad():
                    test_acc, fidelity = get_evaluation(
                        copy_model,
                        victim_model,
                        test_loader,
                        'cuda',
                    )

                logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                    epoch, optimizer.param_groups[0]['lr'], elapsed_time, loss, test_acc,
                    fidelity))

                if fidelity > best_fidelity:
                    torch.save(copy_model.module.state_dict(), output_path)
                    best_fidelity = fidelity

                scheduler.step(epoch)

                    # generate next adv_dataset
            if p < (p_epochs):
                # Jacobian Dataset Augmentation
                new_adv_dataset = jacobian_dataset_augmentation(copy_model, adv_loader=adv_loader, lambda_=lambda_,
                                                            device="cuda", num_classes=num_classes)
                adv_dataset = ConcatDataset([adv_dataset, new_adv_dataset])

    elif args.mea_type == 'df':
        # print("test_data.max", torch.max(test_set[0][0]))
        # print("test_data.min", torch.min(test_set[0][0]))

        # copy_model = copy.deepcopy(victim_model)
        # copy_model.load_state_dict(torch.load(osp.join(supervisor.get_poison_set_dir(), 'extract_pb_model_unlearned_20.pt')))

        copy_model = dfme(copy_model, victim_model, test_loader, num_classes, logger,
                          output_path, supervisor.get_poison_set_dir(args), args.dataset)

        if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'cifar20':
            train_dataset = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                    download=True, transform=data_transform)

        elif args.dataset == 'speech_commands':
            train_dataset = SpeechCommandsDataset(os.path.join('./data', 'speech_commands/speech_commands_v0.02'),
                                                  transform=data_transform_aug)

        if os.path.exists(supervisor.get_poison_set_dir(args)+'/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(args)+'/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(train_dataset), 2500, replace=False) #5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(args)+'/domain_set_indices.np', training_indices)

        domain_dataset = Subset(train_dataset, training_indices)

        adversarial_loader = DataLoader(
            domain_dataset,
            batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs
        )#adversarial loader


        optimizer = torch.optim.SGD(copy_model.parameters(), 0.001, momentum=momentum,
                                    weight_decay=weight_decay)
        if args.dataset == 'speech_commands':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                                   factor=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

        criterion = nn.CrossEntropyLoss().cuda()
        logger.info('-----------Extract Model--------------')
        logger.info('Epoch \t lr \t Time \t TrainLoss \t ACC \t Fidelity')

        best_fidelity = 0.0
        for epoch in range(1, 100 + 1):  #TODO epochs # train backdoored base model
            # scheduler.step(epoch)

            elapsed_time, loss = mea_train(adversarial_loader)

            with torch.no_grad():
                test_acc, fidelity = get_evaluation(
                    copy_model,
                    victim_model,
                    test_loader,
                    'cuda',
                )
                val_atk(args, copy_model)

                logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                    epoch, optimizer.param_groups[0]['lr'], elapsed_time, loss, test_acc,
                    fidelity))

            if fidelity > best_fidelity:
                torch.save(copy_model.module.state_dict(), output_path)
                best_fidelity = fidelity

            if args.dataset == 'speech_commands':
                scheduler.step(metrics=loss)
            else:
                scheduler.step()

    else:
        print('<Undefined model extraction attack> = %s' % args.mea_type)
        raise NotImplementedError('<To Be Implemented> MEA = %s' % args.mea_type)

