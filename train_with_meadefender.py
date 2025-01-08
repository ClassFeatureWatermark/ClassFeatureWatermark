'''codes used to train backdoored models on poisoned dataset
'''
import argparse
import os, sys
import time

from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from MGDA import MGDASolver
from mea_dataset import MixDataset
from mea_mixer import HalfMixer
from tools import generate_dataset, generate_dataloader, val_atk
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default=default_args.parser_default['poison_type'],
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
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
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

parser.add_argument('-lipschiz', type=bool, default=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() *
                        F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class CompositeLoss(nn.Module):
    all_mode = ("cosine", "hinge", "contrastive")

    def __init__(self, rules, simi_factor, mode,
                 size_average=True, *simi_args):
        """
        rules: a list of the attack rules, each element looks like (trigger1, trigger2, ..., triggerN, target)
        """
        super(CompositeLoss, self).__init__()
        self.rules = rules
        self.size_average = size_average
        self.simi_factor = simi_factor

        self.mode = mode
        if self.mode == "cosine":
            self.simi_loss_fn = nn.CosineEmbeddingLoss(*simi_args)
        elif self.mode == "hinge":
            self.pdist = nn.PairwiseDistance(p=1)
            self.simi_loss_fn = nn.HingeEmbeddingLoss(*simi_args)
        elif self.mode == "contrastive":
            self.simi_loss_fn = ContrastiveLoss(*simi_args)


    def forward(self, y_hat, y):

        ce_loss = nn.CrossEntropyLoss()(y_hat, y)

        simi_loss = 0

        for rule in self.rules:
            mask = torch.BoolTensor(size=(len(y),)).fill_(0).cuda()
            for trigger in rule:
                mask |= y == trigger

            if mask.sum() == 0:
                continue

            # making an offset of one element
            y_hat_1 = y_hat[mask][:-1]
            y_hat_2 = y_hat[mask][1:]
            y_1 = y[mask][:-1]
            y_2 = y[mask][1:]

            if self.mode == "cosine":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.cuda())
            elif self.mode == "hinge":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(self.pdist(y_hat_1, y_hat_2), class_flags.cuda())
            elif self.mode == "contrastive":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * 0
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.cuda())

            if self.size_average:
                loss /= y_hat_1.shape[0]

            simi_loss += loss

        return ce_loss, self.simi_factor * simi_loss


if args.trigger is None:
    if args.dataset != 'imagenet':
        args.trigger = config.trigger_default.get(args.poison_type, 'none')
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers.get(args.poison_type, 'none')

all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True


# tools.setup_seed(args.seed)

if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'base')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

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

else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)

if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 0, 'pin_memory': True}
#
# Set Up Poisoned Set
if args.dataset != 'ember' and args.dataset != 'imagenet':
    poison_set_dir = supervisor.get_poison_set_dir(args)
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    print('dataset : %s' % poisoned_set_img_dir)

    poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                     label_path=poisoned_set_label_path, transforms=data_transform if args.no_aug else data_transform_aug)

    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    train_set = datasets.CIFAR10(os.path.join(config.data_dir, 'cifar10'), train=True,
                                 download=True, transform=data_transform)

    # print("train_set", train_set[0])
    # print("poi_set", poisoned_set[0])

    # combined_dataset = ConcatDataset([poisoned_set, train_set])
    # combined_dataset = CombinedDataset(poisoned_set, train_set)


    poisoned_set_loader = torch.utils.data.DataLoader(
        # combined_dataset,
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

elif args.dataset == 'imagenet':
    # TODO

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    print('dataset : %s' % poison_set_dir)
    #
    # poison_indices = torch.load(poison_indices_path)
    #
    # root_dir = '/path_to_imagenet/'
    # train_set_dir = os.path.join(root_dir, 'train')
    # test_set_dir = os.path.join(root_dir, 'val')
    #
    # from utils import imagenet
    # poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, poison_directory=poisoned_set_img_dir,
    #                                          poison_indices = poison_indices, target_class=imagenet.target_class,
    #                                          num_classes=1000)
    #
    # poisoned_set_loader = torch.utils.data.DataLoader(
    #     poisoned_set,
    #     batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

else:
    #TODO

    poison_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    # #stats_path = os.path.join('data', 'ember', 'stats')
    # poisoned_set = tools.EMBER_Dataset( x_path=os.path.join(poison_set_dir, 'watermarked_X.npy'),
    #                                     y_path=os.path.join(poison_set_dir, 'watermarked_y.npy'))
    # print('dataset : %s' % poison_set_dir)
    #
    # poisoned_set_loader = torch.utils.data.DataLoader(
    #     poisoned_set,
    #     batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

#test dataset
if args.dataset != 'ember' and args.dataset != 'imagenet':
    CLASS_A = 0
    CLASS_B = 1
    CLASS_C = 2
    mixer = {
        "Half": HalfMixer()
    }
    poi_set = generate_dataset(dataset=args.dataset, dataset_path=config.data_dir,
                               batch_size=batch_size, split='test',
                               data_transform=data_transform)
    poi_set = MixDataset(dataset=poi_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1,
                         transform=None)  # constructed from test data
    test_set_backdoor_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=batch_size, shuffle=False)

    test_set_loader = generate_dataloader(dataset=args.dataset, dataset_path=config.data_dir,
                                      batch_size=batch_size, split="test",
                                      shuffle=False, drop_last=False,
                                      data_transform=data_transform)

    # Set Up Test Set for Debug & Evaluation
    # test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    # test_set_img_dir = os.path.join(test_set_dir, 'data')
    # test_set_label_path = os.path.join(test_set_dir, 'labels')
    # test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
    #                              label_path=test_set_label_path, transforms=data_transform)
    # test_set_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)
    #
    # # Poison Transform for Testing
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)


# elif args.dataset == 'imagenet':

    # poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)
    #
    # test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
    #              label_file=imagenet.test_set_labels, num_classes=1000)
    # test_set_backdoor = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
    #              label_file=imagenet.test_set_labels, num_classes=1000, poison_transform=poison_transform)
    #
    # test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    # test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))
    #
    # test_set = torch.utils.data.Subset(test_set, test_indices)
    # test_set_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)
    #
    # test_set_backdoor = torch.utils.data.Subset(test_set_backdoor, test_indices)
    # test_set_backdoor_loader = torch.utils.data.DataLoader(
    #     test_set_backdoor,
    #     batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

# else:
    # normalizer = poisoned_set.normal
    #
    # test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    #
    # test_set = tools.EMBER_Dataset(x_path=os.path.join(test_set_dir, 'X.npy'),
    #                                y_path=os.path.join(test_set_dir, 'Y.npy'),
    #                                normalizer = normalizer)
    #
    # test_set_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)
    #
    #
    # backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    # backdoor_test_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X_test.npy'),
    #                                    y_path=None, normalizer = normalizer)
    # backdoor_test_set_loader = torch.utils.data.DataLoader(
    #     backdoor_test_set,
    #     batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

import time
import numpy as np

def get_grads(net, loss):
    params = [x for x in net.parameters() if x.requires_grad]
    grads = list(torch.autograd.grad(loss, params,
                                     retain_graph=True))
    return grads

if __name__ == '__main__':
    # Train Code
    if args.dataset != 'ember':
        model = arch(num_classes=num_classes)
    else:
        model = arch()

    milestones = milestones.tolist()
    model = nn.DataParallel(model)
    model = model.cuda()

    if args.dataset != 'ember':
        print(f"Will save to '{supervisor.get_model_dir(args)}'.")
        if os.path.exists(supervisor.get_model_dir(args)):
            print(f"Model '{supervisor.get_model_dir(args)}' already exists!")

        if args.dataset == 'imagenet':
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()
    else:
        model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options,
                                  'full_base_aug_seed=%d.pt' % args.seed)
        print(f"Will save to '{model_path}'.")
        if os.path.exists(model_path):
            print(f"Model '{model_path}' already exists!")
        criterion = nn.BCELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
        source_classes = [config.source_class]
    else:
        source_classes = None

    train_set_A = []
    train_set_B = []
    Ca = 0
    Cb = 0
    for (img, label) in train_set:
        # print("img, label,", img, ',', label)
        if (label == CLASS_A and Ca <= len(train_set) * 0.1):
            train_set_A.append(img)
            Ca = Ca + 1
        if (Ca == 600):
            break
    # print("A")

    for (img, label) in train_set:
        if (label == CLASS_B and Cb <= len(train_set) * 0.1):
            train_set_B.append(img)
            Cb = Cb + 1
        if (Cb == 600):
            break
    # print("B")

    poi_set_2 = MixDataset(dataset=train_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                           data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1, transform=None)
    train_set_C = []
    Cc = 0
    for (img, label, _) in poi_set_2:
        train_set_C.append(img)
        Cc = Cc + 1
        if (Cc == 600):
            break
    # print("C")

    st = time.time()

    criterion = CompositeLoss(rules=[(CLASS_A, CLASS_B, CLASS_C)], simi_factor=1,
                              mode='contrastive')  # this is semi-supervised learning?
    train_set_A = torch.stack(train_set_A, dim=0)
    train_set_B = torch.stack(train_set_B, dim=0)
    train_set_C = torch.stack(train_set_C, dim=0)
    samples = [train_set_A, train_set_B, train_set_C]
    # scaler = GradScaler()
    for epoch in range(1, epochs + 1):  # train backdoored base model
        start_time = time.perf_counter()

        # Train
        model.train()
        preds = []
        labels = []
        for data, target in tqdm(poisoned_set_loader):

            data = data.cuda()
            target = target.cuda()

            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            # with autocast():
            output = model(data)
            # loss = criterion(output, target)
            loss_A, loss_B = (criterion(output, target))

            with torch.no_grad():
                Sample_A = samples[0]
                Sample_B = samples[1]
                Sample_C = samples[2]

                A_preds = model(Sample_A)
                B_preds = model(Sample_B)

            C_preds = model(Sample_C)

            divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
            ditillation_loss_AC = divergence_loss_fn(F.log_softmax(A_preds, dim=1), F.softmax(C_preds, dim=1)) * 1
            ditillation_loss_BC = divergence_loss_fn(F.log_softmax(B_preds, dim=1), F.softmax(C_preds, dim=1)) * 1
            distillation_loss = ditillation_loss_AC + ditillation_loss_BC

            ori_grads_A = get_grads(model, loss_A)
            ori_grads_B = get_grads(model, loss_B)
            distill_grad = get_grads(model, distillation_loss + loss_B)

            scales = MGDASolver.get_scales(dict(ce1=ori_grads_A, ce2=distill_grad),
                                           dict(ce1=loss_A, ce2=loss_B + distillation_loss),
                                           'loss+', ['ce1', 'ce2'])

            loss = loss_A + scales['ce2'] * (loss_B + distillation_loss)

            if (epoch % 10 == 9):
                loss = loss + 12 * (ditillation_loss_AC + ditillation_loss_BC)
            else:
                loss = loss + 2 * (ditillation_loss_AC + ditillation_loss_BC)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            optimizer.step()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(),
                                                                                                     optimizer.param_groups[
                                                                                                         0]['lr'],
                                                                                                     elapsed_time))
        scheduler.step()

        # Test

        if args.dataset != 'ember':
            if True:
                # if epoch % 5 == 0:
                if args.dataset == 'imagenet':
                    tools.test_imagenet(model=model, test_loader=test_set_loader,
                                        test_backdoor_loader=test_set_backdoor_loader)
                    torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
                else:
                    # tools.test(model=model, test_loader=test_set_loader, poison_test=True,
                    #            poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes, all_to_all=all_to_all)
                    val_atk(args, model)
                    torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
        else:
            tools.test_ember(model=model, test_loader=test_set_loader,
                             backdoor_test_loader=test_set_backdoor_loader)
            torch.save(model.module.state_dict(), model_path)
        print("")

    if args.dataset != 'ember':
        torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
    else:
        torch.save(model.module.state_dict(), model_path)

    if args.poison_type == 'none':
        if args.no_aug:
            torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug.pt')
            torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug_seed={args.seed}.pt')
        else:
            torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug.pt')
            torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug_seed={args.seed}.pt')
