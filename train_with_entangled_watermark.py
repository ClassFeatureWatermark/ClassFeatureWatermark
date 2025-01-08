import argparse
import logging
import os, sys
import time

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tools import val_atk
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler
import numpy as np

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools

def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label in ds:
        classwise_ds[label].append(img)#(img, label)
    return classwise_ds

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default=default_args.parser_default['poison_type'],
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float, required=False,
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
#EWE
parser.add_argument('--source', help='source class of watermark', type=int, default=8)
parser.add_argument('--target', help='target class of watermark', type=int, default=0)
parser.add_argument('--maxiter', help='iter of perturb watermarked data with respect to snnl', type=int, default=10)
parser.add_argument('--w_lr', help='learning rate for perturbing watermarked data', type=float, default=0.01)
parser.add_argument('--t_lr', help='learning rate for temperature', type=float, default=0.1)
parser.add_argument('--temperatures', help='temperature for snnl', nargs='+', type=float, default=torch.tensor([1, 1, 1.]))

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

    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs_wo_w = 100#100
    epochs_w = 10
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

# Set Up Poisoned Set
data_dir = config.data_dir  # directory to save standard clean set
if args.dataset != 'ember' and args.dataset != 'imagenet':
    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    # ])
    train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                 download=True, transform=data_transform_aug)
    classwise_train_data = get_classwise_ds(train_set, num_classes=num_classes)

    # source_data = classwise_train_data[int(args.source)]  # 8
    target_data = classwise_train_data[int(args.target)]  # 0

    # trigger_dataset = []
    # for i in range(len(source_data)):
        # trigger_dataset.append((source_data[i], args.target))
    # trigger_loader = DataLoader(trigger_dataset, batch_size=batch_size, shuffle=True)
    target_dataset = []
    for i in range(len(target_data)):
        target_dataset.append((target_data[i], args.target))
    # poison_set_dir = supervisor.get_poison_set_dir(args)
    # poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    # poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    # poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    # print('dataset : %s' % poisoned_set_img_dir)

    # poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
    #                                  label_path=poisoned_set_label_path,
    #                                  transforms=data_transform if args.no_aug else data_transform_aug)

    # poisoned_set_loader = torch.utils.data.DataLoader(
    #     poisoned_set,
    #     batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)


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

else:
    #TODO ember
    print("Need to implement ember")

    # poison_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    # poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    #
    # stats_path = os.path.join('data', 'ember', 'stats')
    # poisoned_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X.npy'),
    #                                    y_path=os.path.join(poison_set_dir, 'watermarked_y.npy'))
    # print('dataset : %s' % poison_set_dir)
    #
    # poisoned_set_loader = torch.utils.data.DataLoader(
    #     poisoned_set,
    #     batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

if args.dataset != 'ember' and args.dataset != 'imagenet':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # TODO Poison Transform for Testing
    # poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
    #                                                    target_class=config.target_class[args.dataset],
    #                                                    trigger_transform=data_transform,
    #                                                    is_normalized_input=True,
    #                                                    alpha=args.alpha if args.test_alpha is None else args.test_alpha,
    #                                                    trigger_name=args.trigger, args=args)


# elif args.dataset == 'imagenet':
#
#     poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)
#
#     test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
#                                          label_file=imagenet.test_set_labels, num_classes=1000)
#     test_set_backdoor = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
#                                                   label_file=imagenet.test_set_labels, num_classes=1000,
#                                                   poison_transform=poison_transform)
#
#     test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
#     test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))
#
#     test_set = torch.utils.data.Subset(test_set, test_indices)
#     test_set_loader = torch.utils.data.DataLoader(
#         test_set,
#         batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)
#
#     test_set_backdoor = torch.utils.data.Subset(test_set_backdoor, test_indices)
#     test_set_backdoor_loader = torch.utils.data.DataLoader(
#         test_set_backdoor,
#         batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)
#
# else:
#     normalizer = poisoned_set.normal
#
#     test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
#
#     test_set = tools.EMBER_Dataset(x_path=os.path.join(test_set_dir, 'X.npy'),
#                                    y_path=os.path.join(test_set_dir, 'Y.npy'),
#                                    normalizer=normalizer)
#
#     test_set_loader = torch.utils.data.DataLoader(
#         test_set,
#         batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)
#
#     backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
#     backdoor_test_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X_test.npy'),
#                                             y_path=None, normalizer=normalizer)
#     backdoor_test_set_loader = torch.utils.data.DataLoader(
#         backdoor_test_set,
#         batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

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

    criterion = nn.CrossEntropyLoss().cuda()
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'full_base_aug_seed=%d.pt' % args.seed)
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

import time

st = time.time()

# scaler = GradScaler()
args.model = 'full_base_aug_seed=2333-step1-100.pt'
# # train the model without watermarks
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# for epoch in range(1, 10+1):#epoch_wo_w
#     start_time = time.perf_counter()
#     model.train()
#     preds = []
#     labels = []
#     for data, target in tqdm(train_loader):
#         data = data.cuda()
#         target = target.cuda()
#         optimizer.zero_grad()
#
#         # with autocast():
#         output = model(data)
#         loss = criterion(output, target)
#
#         # scaler.scale(loss).backward()
#         # scaler.step(optimizer)
#         # scaler.update()
#
#         loss.backward()
#         optimizer.step()
#
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time
#     print('<Backdoor Training> Train Epoch: {} '
#           '\tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(),
#                                                              optimizer.param_groups[0]['lr'],
#                                                              elapsed_time))
#     scheduler.step()
#
#     # Test
#     if not os.path.exists(supervisor.get_poison_set_dir(args)):
#         os.mkdir(supervisor.get_poison_set_dir(args))
#
#     if args.dataset != 'ember':
#         if True:
#             # if epoch % 5 == 0:
#             if args.dataset == 'imagenet':
#                 tools.test_imagenet(model=model, test_loader=test_set_loader)
#                 torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
#             else:
#                 tools.test(model=model, test_loader=test_set_loader, poison_test=False,
#                            num_classes=num_classes, source_classes=source_classes,
#                            all_to_all=all_to_all)
#                 torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
#     else:
#
#         tools.test_ember(model=model, test_loader=test_set_loader)
#         torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
#     print("")
model.module.load_state_dict(torch.load(supervisor.get_model_dir(args)))
args.model = None

#fgsm_optimize_trigger
# print("fgsm_optimze_trigger")
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
# iter_target_loader = iter(target_loader)
#
# step_list = np.zeros([len(train_loader)])
# final_trigger = []
# for _, (trigger_samples, targets1) in tqdm(enumerate(train_loader)):
#     for epoch in range(args.maxiter):
#         trigger_samples = trigger_samples.cuda().requires_grad_()
#         output = model(trigger_samples) #ewe/activation
#         prediction = torch.unbind(output, dim=1)[args.target]
#         gradient = torch.autograd.grad(prediction, trigger_samples, grad_outputs=torch.ones_like(prediction))[0]
#         trigger_samples = torch.clip(trigger_samples - args.w_lr * torch.sign(gradient[:len(trigger_samples)]), 0, 1)
#         try:
#             target_samples, targets2 = iter_target_loader.next()
#         except:
#             iter_target_loader = iter(target_loader)
#             target_samples, targets2 = iter_target_loader.next()
#         target_samples = target_samples.cuda()
#         targets2 = targets2.cuda()
#
#         batch_data = torch.vstack((trigger_samples, target_samples))
#         w_labels = torch.cat((torch.ones(len(trigger_samples)), torch.zeros(len(target_samples)))).cuda()
#         predictions = model.module.snnl_trigger(batch_data, w_labels, args.temperatures)
#         gradient = torch.autograd.grad(predictions, batch_data,
#                                                grad_outputs=[torch.ones_like(pred) for pred in predictions])[0]
#         trigger_samples = torch.clip(trigger_samples + args.w_lr * torch.sign(gradient[:len(trigger_samples)]),
#                     0, 1)
#
#         for i in range(5):
#             inputs = torch.vstack((trigger_samples, trigger_samples))
#             output = model(inputs)
#             prediction = torch.unbind(output, dim=1)[args.target]
#             gradient = torch.autograd.grad(prediction, inputs,
#                                         grad_outputs=torch.ones_like(prediction))[0]
#             trigger_samples = torch.clip(trigger_samples + args.w_lr * torch.sign(gradient[:len(trigger_samples)]),
#                     0, 1)
#     # final_trigger[batch * len(trigger_samples): (batch + 1) * len(trigger_samples)] = trigger_samples
#     final_trigger.append(trigger_samples.tolist())
# final_trigger = torch.vstack(final_trigger)

if args.dataset != 'ember' and args.dataset != 'imagenet':
    poison_set_dir = supervisor.get_poison_set_dir(args)
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    # poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    print('dataset : %s' % poisoned_set_img_dir)

    poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                     label_path=poisoned_set_label_path,
                                     transforms=data_transform)
                                     # transforms=transforms.Compose([transforms.ToTensor(),]))
    # poisoned_set =
    # target_dataset =
    print("poisoned_set_loader, ", len(poisoned_set))
    print("target_dataset, ", len(target_dataset))
    poisoned_set_loader = DataLoader(
        Subset(poisoned_set, list(range(2500))),
        batch_size=int(batch_size/2), shuffle=True, worker_init_fn=tools.worker_init, **kwargs)
    target_set_loader = DataLoader(
        target_dataset,
        batch_size=int(batch_size/2), shuffle=True)

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=tools.worker_init, **kwargs)

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
iterator = iter(train_loader)
target_iterator = iter(target_set_loader)
args.temperatures = args.temperatures.cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=momentum, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), 0.001)

logger.info('----------- Train Model using Entangled Watermark--------------')
logger.info('Epoch \t lr \t Time \t TrainLoss \t CleanACC \t PoisonACC')
j = 0
for e in tqdm(range(10)): #10
    # w_num_batch = int(len(trigger)/batch_size)
    start_time = time.time()
    for trigger_batch, trigger_target in poisoned_set_loader:
    # for batch_idx in range(w_num_batch):
        try:
            batch, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch, target = next(iterator)
        batch = batch.cuda()
        target = target.cuda()
        pred = model(batch)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        try:
             target_batch, _ = next(target_iterator)
        except StopIteration:
             target_iterator = iter(target_set_loader)
             target_batch, _ = next(target_iterator)

        batch_data = torch.cat((trigger_batch, target_batch)).cuda()
        pred = model(batch_data)
        args.temperatures = args.temperatures.requires_grad_()
        w_labels = torch.cat((torch.ones(int(len(trigger_batch))), torch.zeros(int(len(target_batch))))).cuda()
        snnl = model.module.snnl_trigger(batch_data, w_labels, args.temperatures)
        grad = \
            torch.autograd.grad(snnl, args.temperatures, grad_outputs=[torch.ones_like(s) for s in snnl])[
                0]
        trigger_label = args.target * torch.ones(int(len(pred)), dtype=torch.long).cuda()
        loss = criterion(pred, trigger_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        args.temperatures.data -= args.t_lr * grad

    end_time = time.time()
    #test

    clean_acc, asr, _ = val_atk(args, model)
    logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
        e, optimizer.param_groups[0]['lr'],
        end_time - start_time, loss.item(), clean_acc, asr))

    torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
