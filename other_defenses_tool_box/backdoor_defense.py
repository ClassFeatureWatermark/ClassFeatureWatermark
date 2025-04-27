import logging
from collections import OrderedDict

from torch import nn

import config, os
from mea_net import MEANet
from utils import supervisor
import torch
from torchvision import datasets, transforms
from PIL import Image

class BackdoorDefense():
    def __init__(self, args):
        self.dataset = args.dataset
        if args.dataset == 'gtsrb':
            if args.no_normalize:
                self.data_transform_aug = transforms.Compose([
                    transforms.RandomRotation(15),
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])
                self.data_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor()
                ])
                self.trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                self.normalizer = transforms.Compose([])
                self.denormalizer = transforms.Compose([])
            else:
                self.data_transform_aug = transforms.Compose([
                    transforms.RandomRotation(15),
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
                self.data_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
                self.trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
                self.normalizer = transforms.Compose([
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
                self.denormalizer = transforms.Compose([
                    transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                            (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
                ])
            self.img_size = 32
            self.num_classes = 43
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar10' or args.dataset == 'cifar20':
            if args.no_normalize:
                self.data_transform_aug = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor()
                ])
                self.data_transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                self.trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                self.normalizer = transforms.Compose([])
                self.denormalizer = transforms.Compose([])
            else:
                self.data_transform_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    # # TODO
                    # transforms.RandomRotation(degrees=15),
                    # # 随机改变亮度、对比度和饱和度
                    # transforms.ColorJitter(
                    #     brightness=0.4,  # 亮度变化范围
                    #     contrast=0.4,  # 对比度变化范围
                    #     saturation=0.4,  # 饱和度变化范围
                    #     hue=0.1  # 色调变化范围
                    # ),
                    # 随机仿射变换，包含缩放、旋转、平移
                    # transforms.RandomAffine(
                    #     degrees=10,  # 随机旋转的角度范围
                    #     translate=(0.1, 0.1),  # 随机平移的比例范围
                    #     scale=(0.8, 1.2),  # 随机缩放的比例范围
                    #     shear=10  # 随机剪切的角度范围
                    # ),
                    # 随机应用高斯模糊
                    # transforms.RandomApply([
                    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                    # ], p=0.5),
                    # transforms.RandomGrayscale(p=0.2),

                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                self.data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                self.trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                self.normalizer = transforms.Compose([
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                self.denormalizer = transforms.Compose([
                    transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
                ])            
            self.img_size = 32
            if args.dataset == 'cifar20':
                self.num_classes = 20
            else:
                self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1

        elif args.dataset == 'Imagenette':
            if args.no_normalize:
                self.data_transform_aug = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                ])
                self.data_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])
                self.trigger_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])

                self.normalizer = transforms.Compose([])
                self.denormalizer = transforms.Compose([])

            else:
                self.data_transform_aug = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

                self.data_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                self.trigger_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                self.normalizer = transforms.Compose([
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                self.denormalizer = transforms.Compose([
                    transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                         [1 / 0.229, 1 / 0.224, 1 / 0.225])
                ])

            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 5e-6
            self.learning_rate = 0.01

        elif args.dataset == 'speech_commands':
            self.num_classes = 12
            self.input_channel = 1
            self.shape = torch.Size([1, 32, 32])
            self.data_transform_aug = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.normalizer = transforms.Compose([])
            self.denormalizer = transforms.Compose([])

        elif args.dataset == 'cifar100':
            # print("cifar100 aug")
            print('<To Be Implemented> Dataset = cifar100')
            # exit(0)

        self.poison_type = args.poison_type
        self.poison_rate = args.poison_rate
        self.cover_rate = args.cover_rate
        self.alpha = args.alpha
        self.trigger = args.trigger
        self.target_class = config.target_class[args.dataset]
        self.device = 'cuda'

        if args.dataset !='speech_command' and (args.poison_type != 'IB') and (args.poison_type != 'composite_backdoor'):
            print("args.poison_type", args.poison_type)
            self.poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                            target_class=config.target_class[args.dataset], trigger_transform=self.trigger_transform,
                                                            is_normalized_input=(not args.no_normalize),
                                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                            trigger_name=args.trigger, args=args)
        else:
            self.poison_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
            self.source_classes = [config.source_class]
        else:
            self.source_classes = None

        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trigger_path = os.path.join(config.triggers_dir, args.trigger)
        print('trigger_path:', trigger_path)
        self.trigger_mark = Image.open(trigger_path).convert("RGB")
        self.trigger_mark = trigger_transform(self.trigger_mark).cuda()
        
        trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % args.trigger)
        if os.path.exists(trigger_mask_path): # if there explicitly exists a trigger mask (with the same name)
            print('trigger_mask_path:', trigger_mask_path)
            self.trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            self.trigger_mask = transforms.ToTensor()(self.trigger_mask)[0].cuda() # only use 1 channel
        else: # by default, all black pixels are masked with 0's (not used)
            print('No trigger mask found! By default masking all black pixels...')
            self.trigger_mask = torch.logical_or(torch.logical_or(self.trigger_mark[0] > 0, self.trigger_mark[1] > 0),
                                                 self.trigger_mark[2] > 0).cuda()

        self.poison_set_dir = supervisor.get_poison_set_dir(args)
        model_path = supervisor.get_model_dir(args)
        print("read_model_from", model_path)
        if "mu_" in args.poison_type:
            arch = config.arch[args.poison_type[:3]+args.dataset]
        else:
            arch = config.arch[args.dataset]
        if args.dataset == 'speech_commands':
            self.model = arch(num_classes=self.num_classes, in_channels=1)
        else:
            if '_11' in args.model:
                self.model = arch(num_classes=self.num_classes+1)

            else:
                self.model = arch(num_classes=self.num_classes)

        if os.path.exists(model_path):
            # if args.poison_type == 'meadefender':
            #     self.model = ResNet18(num_classes=self.num_classes)#MEANet(num_classes=self.num_classes)
            #     self.model.load_state_dict(torch.load(model_path)['net_state_dict'])
            # else:
            #     self.model.load_state_dict(torch.load(model_path))
            # state_dict = torch.load(model_path)
            # for k, _ in state_dict.items():
            #     if 'module' in k:
            #         change_k = True
            #     else:
            #         change_k = False
            #     break
            #
            # if change_k:
            #     new_state_dict = OrderedDict()
            #         for k, v in state_dict.items():
            #             new_state_dict[k[7:]] = v
            #     self.model.load_state_dict(new_state_dict)
            # else:
            #     self.model.load_state_dict(torch.load(model_path))

            self.model.load_state_dict(torch.load(model_path))

            # try:
            #     self.model.load_state_dict(torch.load(model_path))
            # except:
            #     print("the weights include 'module")
            #     state_dict = torch.load(model_path)
            #     new_state_dict = OrderedDict()
            #     for k, v in state_dict.items():
            #         new_state_dict[k[7:]] = v
            #     self.model.load_state_dict(new_state_dict)

            #extend to 10 classes
            # forget_class = 4
            # weights = self.model.fc.weight.data
            # bias = self.model.fc.bias.data
            # new_weights = torch.cat([weights[:int(forget_class), :], torch.zeros_like(weights[:1, :]),
            #                          weights[(int(forget_class)):, :]], dim=0)
            # new_bias = torch.cat([bias[:forget_class], torch.zeros_like(bias[:1]), bias[forget_class:]], dim=0)
            # new_output_layer = nn.Linear(weights.size(0), self.num_classes)
            # new_output_layer.weight.data = new_weights
            # new_output_layer.bias.data = new_bias
            # self.model.fc = new_output_layer

            print("Evaluating model '{}'...".format(model_path))
        else:
            print("Model '{}' not found.".format(model_path))
        
        self.model = self.model.cuda()
        self.model.eval()

        
