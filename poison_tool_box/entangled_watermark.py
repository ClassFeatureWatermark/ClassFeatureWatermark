"""
Toolkit for implementing entangled watermarks against mode extractions
[1] Jia, Hengrui, et al. "Entangled watermarks as a defense against model extraction." 30th USENIX security symposium (USENIX Security 21). 2021.
"""
import os
import time

import torch
import random

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm

import config
import supervisor
import tools
import numpy as np

class watermark_generator():
    def __init__(self, args, img_size, dataset, path, target_class=0):
        self.args = args
        self.path = path
        self.source_class = 8
        self.target_class = target_class
        self.epochs_wo_w = 10
        self.epochs_w = 10
        self.maxiter = 10 #'iter of perturb watermarked data with respect to snnl'
        self.w_lr = 0.01
        self.t_lr = 0.1
        self.temperatures = [1, 1, 1]
        self.batch_size = 128
        weight_decay = 1e-4
        if self.args.dataset == 'cifar10':
            self.num_classes = 10
        self.criterion = nn.CrossEntropyLoss().cuda()
        arch = config.arch[self.args.dataset]
        self.model = arch(num_classes=self.num_classes)
        # self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.1, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75])

        # self.train_set = dataset
        self.CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)
        ])
        self.train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                     download=True, transform=data_transform)

        self.img_size = img_size

    def generate_poisoned_training_set(self):
        label_set = []
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        #warmup
        # for epoch in range(1, self.epochs_wo_w + 1):
        #     start_time = time.perf_counter()
        #     self.model.train()
        #     preds = []
        #     labels = []
        #     for data, target in tqdm(train_loader):
        #         data = data.cuda()
        #         target = target.cuda()
        #
        #         data, target = data.cuda(), target.cuda()
        #         self.optimizer.zero_grad()
        #
        #         # with autocast():
        #         output = self.model(data)
        #         loss = self.criterion(output, target)
        #
        #         loss.backward()
        #         self.optimizer.step()
        #
        #     end_time = time.perf_counter()
        #     elapsed_time = end_time - start_time
        #     print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch,
        #                                                                                                  loss.item(),
        #                                                                                                  self.optimizer.param_groups[
        #                                                                                                      0]['lr'],
        #                                                                                                  elapsed_time))
        #     self.scheduler.step()

            # Test
            # if not os.path.exists(supervisor.get_poison_set_dir(self.args)):
            #     os.mkdir(supervisor.get_poison_set_dir(self.args))
            # No Test
            # tools.test(model=self.model, test_loader=test_set_loader, poison_test=False,
            #            num_classes=self.num_classes, source_classes=self.source_classes,
            #            all_to_all=all_to_all)
            # model = 'full_base_aug_seed=2333-step1.pt'
            # torch.save(self.model.module.state_dict(), supervisor.get_poison_set_dir(self.args)+f'\\{model}')
            # print("")
        model = 'full_base_aug_seed=2333-step1-100.pt'
        # model = 'MR_CN_full_base_aug_seed=2333.pt'
        self.model.load_state_dict(torch.load(supervisor.get_poison_set_dir(self.args)+ f'\\{model}'))

        print("fgsm_optimze_trigger")
        classwise_train_data = get_classwise_ds_ewe(self.train_set, num_classes=self.num_classes)

        source_data = classwise_train_data[int(self.source_class)]  # 8
        target_data = classwise_train_data[int(self.target_class)]  # 0

        trigger_dataset = []
        for i in range(len(source_data)):
            trigger_dataset.append((source_data[i], self.target_class))
        trigger_loader = DataLoader(trigger_dataset, batch_size=self.batch_size, shuffle=True)
        target_dataset = []
        for i in range(len(target_data)):
            target_dataset.append((target_data[i], self.target_class))

        target_loader = DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True)
        iter_target_loader = iter(target_loader)

        for batch_idx, (trigger_samples, targets1) in tqdm(enumerate(trigger_loader)):
            for epoch in range(self.maxiter):
                trigger_samples = trigger_samples.cuda().requires_grad_()
                output = self.model(trigger_samples)  # ewe/activation
                prediction = torch.unbind(output, dim=1)[self.target_class]
                gradient = torch.autograd.grad(prediction, trigger_samples, grad_outputs=torch.ones_like(prediction))[0]
                trigger_samples = torch.clip(trigger_samples - self.w_lr * torch.sign(gradient[:len(trigger_samples)]),
                                             0, 1)
                #snnl
                try:
                    target_samples, _ = iter_target_loader.next()
                except:
                    iter_target_loader = iter(target_loader)
                    target_samples, _ = iter_target_loader.next()
                target_samples = target_samples.cuda()
                # targets2 = targets2.cuda()

                batch_data = torch.vstack((trigger_samples, target_samples))
                w_labels = torch.cat((torch.ones(len(trigger_samples)), torch.zeros(len(target_samples)))).cuda()
                predictions = self.model.snnl_trigger(batch_data, w_labels, self.temperatures)
                gradient = torch.autograd.grad(predictions, batch_data,
                                               grad_outputs=[torch.ones_like(pred) for pred in predictions])[0]
                trigger_samples = torch.clip(trigger_samples + self.w_lr * torch.sign(gradient[:len(trigger_samples)]),
                                             0, 1)

                for _ in range(5):
                    inputs = torch.vstack((trigger_samples, trigger_samples))
                    output = self.model(inputs)
                    prediction = torch.unbind(output, dim=1)[self.target_class]
                    gradient = torch.autograd.grad(prediction, inputs,
                                                   grad_outputs=torch.ones_like(prediction))[0]
                    trigger_samples = torch.clip(
                        trigger_samples + self.w_lr * torch.sign(gradient[:len(trigger_samples)]),
                        0, 1)

            # final_trigger[batch * len(trigger_samples): (batch + 1) * len(trigger_samples)] = trigger_samples
            # final_trigger.append(trigger_samples.tolist())
            for i in range(len(trigger_samples)):
                img = trigger_samples[i]
                unnormalize = transforms.Normalize(mean=[-m / s for m, s in zip(self.CIFAR_MEAN, self.CIFAR_STD)],
                                                   std=[1 / s for s in self.CIFAR_STD])
                img = unnormalize(img)
                img_file_name = '%d.png' % (i + batch_idx * self.batch_size)
                img_file_path = os.path.join(self.path, img_file_name)
                save_image(img, img_file_path)
                print('[Generate Poisoned Set] Save %s' % img_file_path)
            if batch_idx == 0:
                label_set = targets1.cpu().numpy().tolist()
            else:
                label_set += targets1.cpu().numpy().tolist()

            #TODO the length of watermark data
            if len(label_set) > 250:
                break

        return torch.LongTensor(label_set)
def get_classwise_ds_ewe(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label in ds:
        classwise_ds[label].append(img)#(img, label)
    return classwise_ds


