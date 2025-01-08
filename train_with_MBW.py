import argparse
import logging
import os, sys
import time

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import mbw_watermarks
import queries
from tools import val_atk
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler
import numpy as np

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class MultiAverageMeter(object):
    def __init__(self, names):
        self.AMs = {name: AverageMeter(name) for name in names}

    def __getitem__(self, name):
        return self.AMs[name].avg

    def reset(self):
        for am in self.AMs.values():
            am.reset()

    def update(self, vals, n=1):
        for name, value in vals.items():
            self.AMs[name].update(value, n=n)

    def __str__(self):
        return ' '.join([str(am) for am in self.AMs.values()])

def loop(model, watermark_model, loader, opt, lr_scheduler, epoch, mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'watermark loss', 'watermark acc'])

    for batch_idx, (images, labels) in enumerate(loader):
        epoch_with_batch = epoch + (batch_idx + 1) / len(loader)
        if lr_scheduler is not None:
            lr_new = lr_scheduler(epoch_with_batch)
            for param_group in opt.param_groups:
                param_group.update(lr=lr_new)

        images = images.to(device)
        labels = labels.to(device)

        # pattern is margin
        #num_sample_fn = lambda x: np.interp([x], [0, max_epoch], [25, 25])[0]
        num_sample = 25 #int(num_sample_fn(epoch))
        if mode == 'train':
            watermark_sample, response = watermark_model(discretize=False, num_sample=num_sample) #it's random sample
            watermark_sample, response = watermark_sample.to(device), response.to(device)
            model.eval()
            for _ in range(5):
                watermark_sample = watermark_sample.detach()
                watermark_sample.requires_grad_(True)
                wm_preds = model(watermark_sample)
                wm_loss = F.cross_entropy(wm_preds, response)
                wm_loss.backward()
                watermark_sample = watermark_sample + watermark_sample.grad.sign() * (1 / 255)
                watermark_sample = torch.clamp(watermark_sample, 0., 1.)
            watermark_sample = watermark_sample.detach()
            model.train()
            # model.zero_grad()
        else:
            watermark_sample, response = watermark_model(discretize=(mode != 'train'))
            watermark_sample, response = watermark_sample.to(device), response.to(device)

        if mode == 'train':
            model.train()
            opt.zero_grad()

        preds = model(images)
        nat_loss = F.cross_entropy(preds, labels, reduction='none')

        wm_preds = model(watermark_sample)
        watermark_acc = (wm_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
        watermark_loss = addvar * F.cross_entropy(wm_preds, response, reduction='none')
        loss = torch.cat([nat_loss, watermark_loss]).mean()
        # loss = nat_loss.mean()
        nat_acc = (preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()

        if mode == 'train':
            loss.backward()
            opt.step()

        meters.update({
            'nat loss': nat_loss.mean().item(),
            'nat acc': nat_acc.item(),
            'watermark loss': watermark_loss.mean().item(),
            'watermark acc': watermark_acc.item()
        }, n=images.size(0))

        # if batch_idx % 100 == 0 and mode == 'train':
        #     logger.info('=====> {} {}'.format(mode, str(meters)))

    # logger.info("({:3.1f}%) Epoch {:3d} - {} {}".format(100 * epoch / max_epoch, epoch, mode.capitalize().ljust(6),
    #                                                     str(meters)))

    return meters

# def get_classwise_ds(ds, num_classes):
#     classwise_ds = {}
#     for i in range(num_classes):
#         classwise_ds[i] = []
#
#     for img, label in ds:
#         classwise_ds[label].append(img)#(img, label)
#     return classwise_ds

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='MBW',
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float, required=False,
                    default=0.2)
                    # choices=default_args.parser_choices['poison_rate'])
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
#MBW
parser.add_argument('-nm', "--num-mixup",
                    type=int,
                    help='# of mixup',
                    default=1)
parser.add_argument('-v', "--variable",
        type=float,
        default=0.1)
parser.add_argument('-nq', "--num-watermark",
                    type=int,
                    help='# of watermarks',
                    default=10)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

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
    out_path = os.path.join(out_path, '%s_%s.out' % (
    supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
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
    watermark_size = (3, 32, 32)
    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    # epochs_wo_w = 100#100
    epochs_w = 200 #100]
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'gtsrb':
    watermark_size = (3, 32, 32)
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

# Set Up Poisoned Set
data_dir = config.data_dir  # directory to save standard clean set
if args.dataset != 'ember' and args.dataset != 'imagenet':
    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    # ])
    # train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
    #                              download=True, transform=data_transform)
    train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                 download=True, transform=data_transform) #data_transform_aug
    test_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
                                download=True, transform=data_transform)

elif args.dataset == 'imagenet':
    # TODO imagenet
    print("Need to implement imagenet")

    # poison_set_dir = supervisor.get_poison_set_dir(args)
    # poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    # poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    # print('dataset : %s' % poison_set_dir)
    #
    # poison_indices = torch.load(poison_indices_path)
    #
    # root_dir = '/path_to_imagenet/'
    # train_set_dir = os.path.join(root_dir, 'train')
    # test_set_dir = os.path.join(root_dir, 'val')
    #
    # from utils import imagenet
    #
    # poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, poison_directory=poisoned_set_img_dir,
    #                                          poison_indices=poison_indices, target_class=imagenet.target_class,
    #                                          num_classes=1000)
    #
    # poisoned_set_loader = torch.utils.data.DataLoader(
    #     poisoned_set,
    #     batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)


model = arch(num_classes=num_classes)

milestones = milestones.tolist()
model = nn.DataParallel(model)
model = model.cuda()
#num_watermark seems the batch
watermark =queries.StochasticImageQuery(query_size=(args.num_watermark, *watermark_size),
                                response_size=(args.num_watermark,), query_scale=255, response_scale=num_classes)
           #mbw_watermarks.ImageMBW(watermark_size=(args.num_watermark, *watermark_size),
                                    # response_size=(args.num_watermark,), watermark_scale=255,
                                    # response_scale=num_classes)#0-9
# watermark = nn.DataParallel(watermark)
# watermark = watermark.cuda()
device = 'cuda'
# watermark_sample = (torch.randint(0, 255, (args.num_watermark, *watermark_size)).float() / torch.tensor(255).float()).to(device)
# response = torch.randint(0, num_classes, [args.num_watermark]).to(device)

if args.dataset != 'ember':
    print(f"Will save to '{supervisor.get_model_dir(args)}'.")
    if os.path.exists(supervisor.get_model_dir(args)):
        print(f"Model '{supervisor.get_model_dir(args)}' already exists!")

    criterion = nn.CrossEntropyLoss().cuda()
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'full_base_aug_seed=%d.pt' % args.seed)
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")
    criterion = nn.BCELoss().cuda()

if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = [config.source_class]
else:
    source_classes = None

import time

# if args.dataset != 'ember' and args.dataset != 'imagenet':
#     # Set Up Test Set for Debug & Evaluation
#     test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
#     test_set_img_dir = os.path.join(test_set_dir, 'data')
#     test_set_label_path = os.path.join(test_set_dir, 'labels')
#     test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
#                                  label_path=test_set_label_path, transforms=data_transform)

if not os.path.exists(os.path.join(supervisor.get_poison_set_dir(args))):
    os.makedirs(os.path.join(supervisor.get_poison_set_dir(args)))

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(supervisor.get_poison_set_dir(args), f'output_{args.dataset}.log')),
            logging.StreamHandler()
    ])
logger.info(args)

criterion = nn.CrossEntropyLoss()

if args.dataset in ['cifar10', 'cifar100']:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
    scheduler = lambda t: np.interp([t], [0, 100, 100, 150, 150, 200],
                                       [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]

elif args.dataset == 'svhn':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
    scheduler = lambda t: np.interp([t],
                                       [0, 100, 100, 150, 150, 200],
                                       [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]

# optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

args.model = None

logger.info('----------- Train Model using Margin-Based Watermark--------------')
logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainCleanAcc \t CleanACC \t PoisonACC')
st = time.time()
best_val_nat_acc = 0
best_val_watermark_acc = 0

for epoch in range(epochs_w):
    start_time = time.time()
    model.train()
    watermark.train()
    train_meters = loop(model, watermark, train_loader, optimizer, scheduler, epoch, mode='train', device=device, addvar=args.variable)

    # scheduler.step()

    #TODO save the model

    with torch.no_grad():
        model.eval()
        watermark.eval()
        val_meters = loop(model, watermark, test_loader, optimizer, scheduler, epoch, mode='test', device=device, addvar=args.variable)

        # if (epoch + 1) % 20 == 0:
        #     torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
        #     torch.save(watermark, os.path.join(supervisor.get_poison_set_dir(args), 'watermark.pt'))

        if best_val_nat_acc <= val_meters['nat acc']:
            torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
            best_val_nat_acc = val_meters['nat acc']

        if best_val_watermark_acc <= val_meters['watermark acc']:
            torch.save(watermark, os.path.join(supervisor.get_poison_set_dir(args), 'watermark.pt'))
            torch.save(watermark.query, os.path.join(supervisor.get_poison_set_dir(args), 'watermark_query.pt'))
            best_val_watermark_acc = val_meters['watermark acc']

    end_time = time.time()

    #test
    logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
        epoch, optimizer.param_groups[0]['lr'],
        end_time - start_time, train_meters['nat loss'], train_meters['nat acc'],  val_meters['nat acc'], val_meters['watermark acc']))

    # logger.info("=" * 100)
    # logger.info("Best valid nat acc   : {:.4f}".format(best_val_nat_acc))
    # logger.info("Best valid watermark acc : {:.4f}".format(best_val_watermark_acc))
