"""
Implementation of I-BAU defense
I-BAU: Adversarial-Unlearning-of-Backdoors-via-Implicit-Hypergradient
"""
import sys, os
from tkinter import E

from EmbedModule import Embbed
from U_Net_Zoo import U_Net

EXT_DIR = ['..']
for DIR in EXT_DIR:
    if DIR not in sys.path: sys.path.append(DIR)

import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import config
import torch.optim as optim
import time
import datetime
from tqdm import tqdm
from tools import val_atk
from other_defenses_tool_box import BackdoorDefense

from utils import supervisor
import random
import torch.nn.functional as F
from torch.autograd import grad

# I-BAU!
class I_BAU_Patch(BackdoorDefense):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.optim = 'SGD'#'Adam'
        self.lr = 0.0005#TODO 0.1 for badnet #0.001
        self.norm_para = 0.001#TODO 0.001

        self.n_rounds = 10#10
        self.K = 5

        if args.dataset == 'cifar10':
            self.batch_size = 200
            self.ratio = 0.05
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True)
        elif args.dataset == 'Imagenette':
            self.batch_size = 16
            self.ratio = 0.1
            full_train_set = datasets.ImageFolder('./data/imagenette2-160/train')
        self.clean_data = DatasetCL(self.ratio, full_dataset=full_train_set, transform=self.data_transform_aug)
        self.clean_loader = DataLoader(self.clean_data, batch_size=self.batch_size, shuffle=True)

        # arch = config.arch[self.args.dataset]
        # self.model = arch(num_classes=self.num_classes)
        # s_model_path = supervisor.get_model_dir(self.args)
        # checkpoint = torch.load(s_model_path)
        # self.model.load_state_dict(checkpoint)
        # self.model = self.model.to(self.device)

    def detect(self):
        # if self.optim == 'SGD':
        #     outer_opt = torch.optim.SGD(self.model.parameters(),
        #                                 lr=self.lr)
        # elif self.optim == 'Adam':
        #     outer_opt = torch.optim.Adam(self.model.parameters(),
        #                                  lr=self.lr)
        outer_opt = torch.optim.SGD(self.model.parameters(),
                                    lr=self.lr,  # self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)

        print("Original Poisoned Models")
        val_atk(self.args, self.model)

        for round in range(self.n_rounds):
            batch_pert = torch.rand_like(self.clean_data[0][0].unsqueeze(0), requires_grad=True, device=self.device)
            #torch.zeros_like(self.clean_data[0][0].unsqueeze(0), requires_grad=True, device=self.device)
            with torch.no_grad():
                batch_pert.mul_(0.1)#0.1
            batch_opt = torch.optim.Adam(params=[batch_pert], lr=0.1)#0.1 for badnet#10? SGD
            for clean_round in range(5):
                for images, labels in self.clean_loader:
                    images = images.to(self.device)
                    # labels = labels.to(self.device)
                    # revers_trigger = ReverseNet(images)
                    # # print("revers_trigger", revers_trigger.shape)
                    # revers_trigger = EmbbedNet(revers_trigger[:, :3 * self.num_classes, :, :],
                    #                            revers_trigger[:, 3 * self.num_classes: 6 * self.num_classes, :, :])
                    # revers_trigger = (revers_trigger) / 255
                    relabels = torch.randint(0, self.num_classes, [len(images)]).cuda()
                    # relabels_expanded = relabels.view(-1, 1, 1, 1, 1).expand(-1, 1, 3, 32, 32)
                    # batch_pert = revers_trigger.reshape(-1, self.num_classes, 3, 32, 32)
                    # batch_pert = torch.gather(batch_pert, 1, relabels_expanded)
                    # batch_pert = batch_pert.reshape(-1, 3, 32, 32)

                    ori_lab = torch.argmax(self.model.forward(images), axis=1).long()
                    # relabels = torch.randint(0, self.num_classes, [len(images)])

                    per_logits = self.model.forward(images + batch_pert)
                    loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
                    # loss = F.cross_entropy(per_logits, relabels, reduction='mean')

                    loss_regu = torch.mean(-loss) + self.norm_para* torch.pow(torch.norm(batch_pert), 2)
                    # loss_regu = loss + 0.001 * torch.pow(torch.norm(batch_pert), 2)
                    batch_opt.zero_grad()
                    # optimizer_rev.zero_grad()
                    loss_regu.backward(retain_graph=True)
                    batch_opt.step()
                    # optimizer_rev.step()

            # l2-ball
            # pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
            # pert = batch_pert
            # unlearn step
            # for batchnum in range(len(self.images_list)):
            for images, labels in self.clean_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # revers_trigger = ReverseNet(images)
                # revers_trigger = EmbbedNet(revers_trigger[:, :3 * self.num_classes, :, :],
                #                            revers_trigger[:, 3 * self.num_classes: 6 * self.num_classes, :, :])
                # revers_trigger = (revers_trigger) / 255
                # relabels = torch.randint(0, self.num_classes, [len(images)]).cuda()
                # relabels_expanded = relabels.view(-1, 1, 1, 1, 1).expand(-1, 1, 3, 32, 32)
                # batch_pert = revers_trigger.reshape(-1, self.num_classes, 3, 32, 32)
                # batch_pert = torch.gather(batch_pert, 1, relabels_expanded)
                # batch_pert = batch_pert.reshape(-1, 3, 32, 32)

                # hparams = list(self.model.parameters())
                # params = [w.detach().requires_grad_(True) for w in pert]
                outer_opt.zero_grad()
                o_loss = self.loss_outer(batch_pert, images, labels)#params
                o_loss.backward()
                # grad_outer_w = grad(o_loss, params, retain_graph=True)
                # grad_outer_hparams = grad(o_loss, hparams, retain_graph=True)

                # grads = grad(self.loss_inner(params, images, labels), params, create_graph=True)
                # w_mapped = [w - 0.1 * g for w, g in zip(params, grads)]

                # vs = [torch.zeros_like(w) for w in params]
                # vs_vec = torch.cat([xx.reshape([-1]) for xx in vs])
                # for k in range(self.K):
                #     vs_prev_vec = vs_vec
                    # vs = grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

                    # vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
                    # vs_vec = torch.cat([xx.reshape([-1]) for xx in vs])
                    # tol = 1e-10
                    # if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
                    #     break

                # grads = grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
                # grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]
                #
                #update grads
                # for l, g in zip(hparams, grads):
                #     if l.grad is None:
                #         l.grad = torch.zeros_like(l)
                #     if g is not None:
                #         l.grad += g

                outer_opt.step()

            #test
            print(f"Round:[{round}]")
            val_atk(self.args, self.model)

        self.args.model = 'I_BAU_' + self.args.model
        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))

    # def loss_inner(self, perturb, images, labels):
    #     # for images, labels in self.clean_loader:
    #     #     images = images.to(self.device)
    #     #     labels = labels.to(self.device)
    #     #     per_img = images + perturb[0]
    #     #     per_logits = self.model.forward(per_img)
    #     #     loss = F.cross_entropy(per_logits, labels, reduction='none')
    #     #     loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(perturb[0]), 2)
    #     #     break
    #     per_img = images + perturb[0]
    #     per_logits = self.model.forward(per_img)
    #     loss = F.cross_entropy(per_logits, labels, reduction='none')
    #     loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(perturb[0]), 2)
    #
    #     return loss_regu

    def loss_outer(self, perturb, images, labels):
        portion = 0.05#0.01
        patching = torch.zeros_like(images, device=self.device)
        number = images.shape[0]
        rand_idx = random.sample(list(np.arange(number)), int(number * portion))
        patching[rand_idx] = perturb[0] # only limited position is perturbed
        unlearn_imgs = images + patching
        # unlearn_imgs =  images + perturb
        logits = self.model(unlearn_imgs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        # imgs = torch.cat((images, images+perturb[0]), dim=0)
        # labs = torch.cat((labels, labels), dim=0)
        # logits = self.model(imgs)#unlearn_imgs
        # criterion = nn.CrossEntropyLoss()
        # loss = criterion(logits, labs)#labels
        return loss

class DatasetCL(Dataset):
    def __init__(self, ratio, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
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