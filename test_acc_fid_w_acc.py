'''
codes used to train ood-class models on poisoned dataset
'''
import argparse
import logging
import os, sys
import random
import time

from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, WeightedRandomSampler
from torchvision.datasets import CIFAR100
from tqdm import tqdm

import queries
from cifar20 import CIFAR20, OODDataset, CombinedDataset, OodDataset, Dataset4List, OodDataset_from_Dataset
from imagenet import Imagenet64
from mu_utils import get_classwise_ds, get_one_classwise_ds, build_retain_sets_in_unlearning, build_retain_sets_sc, \
    build_ood_sets_sc
from speech_commands import SpeechCommandsDataset, CLASSES
from speech_transforms import ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, ToSTFT, StretchAudioOnSTFT, \
    FixSTFTDimension, TimeshiftAudioOnSTFT, ToMelSpectrogramFromSTFT, DeleteSTFT, ToTensor, LoadAudio, ToMelSpectrogram
from tools import val_atk
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler
import os.path as osp
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='ood_class')
parser.add_argument('-poison_rate', type=float,  required=False,
                    default=0.00)
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
parser.add_argument('-model', type=str)
parser.add_argument('-victim_model', type=str, default='model_unlearned_10.pt')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools
import time
import numpy as np

if args.trigger is None:
    if args.dataset != 'imagenet':
        args.trigger = config.trigger_default.get(args.poison_type, 'none')
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers(args.poison_type, 'none')

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

elif args.dataset == 'speech_commands':
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

    valid_feature_transform = transforms.Compose(
        [ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    data_transform = transforms.Compose([LoadAudio(),
                                         FixAudioLength(),
                                         valid_feature_transform])
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

milestones = milestones.tolist()

if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 0, 'pin_memory': True}

def valid_model(model, victim_model, MBW=None):
    model.eval()
    victim_model.eval()
    # correct_by_class = {i: 0 for i in range(num_classes+1)}
    # total_by_class = {i: 0 for i in range(num_classes+1)}
    num = 0
    num_non_target = 0
    num_clean_correct = 0
    num_poison_eq_poison_label = 0
    num_poison_eq_poison_label_train = 0
    num_non_target_train = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.cuda(), label.cuda()  # train set batch
            output = model(data)
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            num_clean_correct += pred.eq(label).sum().item()
            num += len(label)

        predictions1 = []
        predictions2 = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                outputs2 = victim_model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                victim_pred = outputs2.argmax(dim=1)
                # num_clean_correct += pred.eq(label).sum().item()
                # num += len(label)

                predictions1.append(pred.cpu())
                predictions2.append(victim_pred.cpu())
            predictions1 = torch.cat(predictions1).flatten()
            predictions2 = torch.cat(predictions2).flatten()
            # print("pred1-pred2", sum(predictions1-predictions2))

            fidelity = accuracy_score(predictions1.numpy(), predictions2.numpy()) * 100.

        if MBW is not None:
            wm_data, wm_label = MBW[0].cuda(), MBW[1].cuda()  # train set batch
            output_wm = model(wm_data)
            pred_wm = output_wm.argmax(dim=1)  # get the index of the max log-probability
            num_poison_eq_poison_label += pred_wm.eq(wm_label).sum().item()
            num_non_target += len(wm_label)
        else:
            for _, (wm_data, wm_label) in enumerate(poisoned_loader):
                wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
                output_wm = model(wm_data)
                pred_wm = output_wm.argmax(dim=1)  # get the index of the max log-probability
                num_poison_eq_poison_label += pred_wm.eq(wm_label).sum().item()
                num_non_target += len(wm_label)

    clean_acc = num_clean_correct / num
    asr = num_poison_eq_poison_label / num_non_target

    # for i in range(num_classes+1):
    #     if total_by_class[i] > 0:
    #         accuracy = correct_by_class[i] / total_by_class[i]
    #         print(f"Class {i} Accuracy {correct_by_class[i]}/{total_by_class[i]}: {accuracy * 100:.4f}%")
    #     else:
    #         print(f"Class {i} Accuracy: N/A (no samples)")

    return clean_acc*100, fidelity, asr*100


def val_atk_for_backdoor(model, victim_model, poison_transform):
    model.eval()
    victim_model.eval()
    num = 0
    num_non_target = 0
    num_clean_correct = 0
    num_poison_eq_poison_label = 0
    num_poison_eq_clean_label = 0

    num_fidelity = 0
    acr = 0  # attack correct rate
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):

            data, label = data.cuda(), label.cuda()  # train set batch
            outputs1 = model(data)
            outputs2 = victim_model(data)
            pred = outputs1.argmax(dim=1)  # get the index of the max log-probability
            victim_pred = outputs2.argmax(dim=1)

            num_clean_correct += pred.eq(label).sum().item()

            num_fidelity += pred.eq(victim_pred).sum().item()
            num += len(label)

            data, poison_label = poison_transform.transform(data, label)
            poison_output = model(data)
            poison_pred = poison_output.argmax(dim=1)  # get the index of the max log-probability
            this_batch_size = len(poison_label)
            for bid in range(this_batch_size):
                if label[bid] != poison_label[bid]:  # samples of non-target classes
                    num_non_target += 1
                    if poison_pred[bid] == poison_label[bid]:
                        num_poison_eq_poison_label += 1
                    if poison_pred[bid] == label[bid]:
                        num_poison_eq_clean_label += 1
                else:
                    if poison_pred[bid] == label[bid]:
                        num_poison_eq_clean_label += 1

    clean_acc = num_clean_correct / num
    asr = num_poison_eq_poison_label / num_non_target
    fid = num_fidelity / num
    return clean_acc, fid, asr


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
                os.path.join(model_path, f'compre_evaluate_{args.dataset}_{args.model}.log')),
            logging.StreamHandler()
        ])
    logger.info(args)
    logger.info('Epoch \t Clean Acc \t Fidelity \t W Acc')
    if args.dataset == 'Imagenet_21':
        test_set_imagenet = Imagenet64(root=osp.join('./data', 'Imagenet64_subset-21'), transforms=data_transform,
                                       train=False)

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

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    victim_model_path = supervisor.get_poison_set_dir(args) + '/{}'.format(args.victim_model)

    if '11' in args.victim_model:
        victim_model = arch(num_classes=num_classes+1)
    else:
        victim_model = arch(num_classes=num_classes)

    victim_model.load_state_dict(torch.load(victim_model_path))
    logger.info('evaluate victim_model from {}'.format(victim_model_path))
    victim_model = nn.DataParallel(victim_model)
    victim_model = victim_model.cuda()
    victim_model.eval()

    model_path = supervisor.get_model_dir(args)

    if '_11' in args.model:
        model = arch(num_classes=num_classes+1)
    else:
        model = arch(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))

    # model_path = supervisor.get_model_dir(args)
    # model = torch.load(model_path, map_location='cpu')

    logger.info('evaluate model from {}'.format(model_path))
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    if args.poison_type == 'MBW':
        watermark = queries.StochasticImageQuery(query_size=(10, 3, 32, 32),
                                                 response_size=(10,), query_scale=255, response_scale=num_classes)
        # watermark.query = torch.load(os.path.join(supervisor.get_poison_set_dir(args), 'watermark_query.pt'))
        watermark = torch.load(os.path.join(supervisor.get_poison_set_dir(args), 'watermark.pt'))
        watermark.eval()
        wm_samples, wm_response = watermark(discretize=True)

        with torch.no_grad():
            clean_acc, fidelity, w_acc = valid_model(model, victim_model, MBW=[wm_samples, wm_response])
            logger.info('{} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                0, clean_acc, fidelity, w_acc))
    elif args.poison_type == 'blend' or args.poison_type == 'badnet':
        poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                           target_class=config.target_class[args.dataset],
                                                           trigger_transform=data_transform,
                                                           is_normalized_input=(not args.no_normalize),
                                                           alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                           trigger_name=args.trigger, args=args)
        with torch.no_grad():
            clean_acc, fidelity, w_acc = val_atk_for_backdoor(model, victim_model, poison_transform)
            logger.info('{} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                0, clean_acc, fidelity, w_acc))
    else: #EWE, meadefender
        poison_set_dir = supervisor.get_poison_set_dir(args)
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
        poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
        # poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
        if args.dataset == 'speech_commands' or args.dataset == 'ag_news':
            ood_train_dataset = SpeechCommandsDataset(
                os.path.join('./data', 'speech_commands/speech_commands_v0.01_ood'),
                transform=data_transform)
            poisoned_set = build_ood_sets_sc(ood_train_dataset, num_classes)
        else:
            print('dataset : %s' % poisoned_set_img_dir)
            poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                             label_path=poisoned_set_label_path,
                                             transforms=data_transform)  # if args.no_aug else data_transform_aug

        if args.poison_type == 'sub_ood_class':
            target_label = 0
            poisoned_set = OodDataset_from_Dataset(dataset=poisoned_set, label=target_label)
        poisoned_loader = DataLoader(
            poisoned_set,
            batch_size=int(batch_size), shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

        with torch.no_grad():
            clean_acc, fidelity, w_acc = valid_model(model, victim_model)
            logger.info('{} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                0, clean_acc, fidelity, w_acc))

        #TODO heat map of poisoned loader

        # def grad_cam(model, image, feature_layer):
        #     model.eval()
        #
        #     feature_gradients = []
        #
        #     def save_gradient(module, input, output):
        #         def hook_function(grad):
        #             feature_gradients.append(grad)
        #
        #         return hook_function
        #
        #     handle = feature_layer.register_full_backward_hook(save_gradient)
        #
        #     # Forward pass
        #     features = feature_layer(image)
        #     output = model(image)
        #
        #     # Get the index of the highest score
        #     _, pred = output.max(dim=1)
        #
        #     # Backward pass
        #     model.zero_grad()
        #     output[0, pred].backward()
        #
        #     # Pool the gradients across the channels
        #     gradients = feature_gradients[0][0]
        #     pooled_gradients = torch.mean(gradients, dim=[0, 1, 2])
        #
        #     # Weight the channels by corresponding gradients
        #     for i in range(features.shape[1]):
        #         features[0, i, ...] *= pooled_gradients[i]
        #
        #     # Average the channels of the features
        #     heatmap = torch.mean(features, dim=1).squeeze().cpu().detach()
        #
        #     # Relu on top of the heatmap
        #     heatmap = np.maximum(heatmap, 0)
        #
        #     # Normalize the heatmap
        #     heatmap /= torch.max(heatmap)
        #     handle.remove()
        #     return heatmap.numpy()
        #
        # def show_heatmap(heatmap, img_tensor, alpha=0.5):
        #     if img_tensor.dim() == 3:
        #         img_tensor = img_tensor.permute(1, 2, 0)
        #     img_tensor = img_tensor.cpu()
        #     plt.imshow(img_tensor.numpy())
        #
        #     plt.imshow(heatmap, alpha=alpha, cmap='jet')
        #     plt.axis('off')
        #     plt.show()
        #
        # image = poisoned_set[0][0]
        # feature_layer = model.layer4[-1]
        # heatmap = grad_cam(model, image, feature_layer)
        # show_heatmap(heatmap, image)


