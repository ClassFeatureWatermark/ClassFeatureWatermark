import os
import sys
import time
import copy
import argparse
import numpy as np

import torch
from ranger import Ranger
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from torchvision.utils import save_image
import warnings
from . import BackdoorDefense
from tqdm import tqdm
from .tools import generate_dataloader, AverageMeter, accuracy, val_atk
import config
from utils import supervisor

Print_level = 1

class SEAM(BackdoorDefense):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_classes = 10
        self.batch_size = 64

        # 5% of the clean train set
        if args.dataset == 'cifar10':
            self.lr = 0.1  # 0.01, ;0.1 in orginal
            self.ratio = 0.05  # 0.05 # ratio of training data to use
            self.full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True,
                                                   transform=self.data_transform, download=True)
        else:
            raise NotImplementedError()

        if os.path.exists(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np'):
            training_indices = torch.load(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np')
        else:
            training_indices = np.random.choice(len(self.full_train_set), 2500, replace=False)  # 5000 for CIFAR-10
            np.save(supervisor.get_poison_set_dir(self.args) + '/domain_set_indices.np', training_indices)
        self.train_data = Subset(self.full_train_set, training_indices)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=self.batch_size,
                                               split='std_test',
                                               shuffle=False,
                                               drop_last=False,
                                               data_transform=self.data_transform)
    def run_seam(self):
        total_time = 0
        time_start = time.time()

        # Step 1: Forgetting
        # Set forgetting lr=1e-2
        # lr = 3e-2
        criterion = torch.nn.CrossEntropyLoss()

        lr = 3e-4 #TODO finetune this para
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


        num_classes = self.num_classes
        acc_bound = 0.2
        # acc_bound = 0.1

        if self.args.dataset == 'cifar10' or self.args.dataset == 'Imagenette':
            fg_rate = 0.05
        elif self.args.dataset == 'gtsrb':
            fg_rate = 0.8

        forget_set = self.train_data
        forget_set = FinetuneDataset(forget_set, num_classes=num_classes, data_rate=fg_rate)
        forget_loader = DataLoader(dataset=forget_set, batch_size=self.batch_size, shuffle=True)

        recover_set = self.train_data
        recover_loader = DataLoader(dataset=recover_set, batch_size=self.batch_size, shuffle=True)

        forget_step = 0
        while True:
            forget_step += 1
            self.model.train()
            acc = []
            for step, (x_batch, y_batch) in enumerate(forget_loader):
                # Randomly relabel y_batch
                y_forget = torch.randint(0, num_classes, y_batch.shape).cuda()
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_forget)
                loss.backward()
                optimizer.step()

                y_pred = torch.max(output.data, 1)[1]
                acc.append((y_pred == y_batch))

                if forget_step >= 100:
                    break

            acc = torch.cat(acc, dim=0)
            n_forget = acc.shape[0]
            acc = acc.sum().item() / n_forget
            print(f'Forgetting --- Step: {forget_step}, ACC: {acc * 100:.2f}% ({n_forget})')

            if acc < acc_bound:
                print('Forgot enough samples')
                break

        # Step 2: Recovering
        epochs = 10
        for epoch in range(epochs):
            self.model.train()
            if epoch > 0 and epoch % 2 == 0:
                lr /= 10
                lr *= 10

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            for step, (x_batch, y_batch) in enumerate(recover_loader):
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

                optimizer.zero_grad()

                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 2 == 0:
                time_end = time.time()
                self.model.eval()
                with torch.no_grad():
                    val_atk(self.args, self.model)

                if Print_level > 0:
                    sys.stdout.write('epoch: {:2}/{}, lr: {:.4f} - {:.2f}s, '
                                     .format(epoch + 1, epochs, lr, time_end - time_start) )
                    sys.stdout.flush()

                total_time += (time_end - time_start)
                time_start = time.time()

        return total_time

    def detect(self):
        val_atk(self.args, self.model)
        total_time = self.run_seam()
        self.args.model = 'SEAM_' + self.args.model
        print("Will save the model to: ", supervisor.get_model_dir(self.args))
        torch.save(self.model.state_dict(), supervisor.get_model_dir(self.args))  # TODO


##############################################################################
# Main
##############################################################################
class FinetuneDataset(Dataset):
    def __init__(self, dataset, num_classes, data_rate=1.0):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

        # Randomly select data_rate of the dataset
        n_data = len(dataset)
        n_single = int(n_data * data_rate / num_classes)
        self.n_data = n_single * num_classes

        # Evenly select data_rate of the dataset
        cnt = [n_single for _ in range(num_classes)]

        self.indices = np.random.choice(n_data, n_data, replace=False)

        self.data = []
        self.targets = []
        for i in self.indices:
            img, lbl = dataset[i]

            if cnt[lbl] > 0:
                self.data.append(img)
                self.targets.append(lbl)
                cnt[lbl] -= 1

    def __getitem__(self, index):
        img, lbl = self.data[index], self.targets[index]
        return img, lbl

    def __len__(self):
        return self.n_data