"""
Toolkit for implementing UAE black-box watermark
Zhu, Hongyu, et al. "Reliable Model Watermarking: Defending Against Theft without Compromising on Evasion." arXiv preprint arXiv:2404.13518 (2024).
"""
import os
import time

import torch
import random

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import config
import supervisor
import tools
import numpy as np
import torch.nn.functional as F

def positional_embedding_creator(num_step: int, pos_dim: int):
    matrix = torch.zeros(num_step, pos_dim)
    for i in range(num_step):
        for j in range(0, pos_dim, 2):
            matrix[i, j] = np.sin(i/(10000**(j/pos_dim)))
            if(j+1<pos_dim):
                matrix[i, j+1] = np.cos(i/(10000**(j/pos_dim)))

    return matrix

class watermark_generator():
    def __init__(self, args, path):
        self.args = args
        self.path = path

        if self.args.dataset == 'cifar10':
            self.num_classes = 10
        self.target_class = 1 # for simplify
        # downlaod target_model from 'EWE-step1' or 'clean model'
        self.target_model = config.arch[self.args.dataset]()
        self.target_model.load_state_dict(
        torch.load(
            './poisoned_train_set/cifar10/entangled_watermark_0.050_poison_seed=0/full_base_aug_seed=2333-step1-100.pt'))
        self.target_model = nn.DataParallel(self.target_model)
        self.target_model = self.target_model.cuda()
        self.target_model.eval()

        self.unet = torch.load("./downloaded_model/diffusion_CIFAR10.pth").cuda()
        # unet = nn.DataParallel(unet).to(device)
        print("unet is downloaded")

        self.batch_size = 128
        self.watermark_num = 100
        self.watermark_set = []
        self.round = 5#15#used for generating watermark samples
        self.data_transform = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

    def generate_poisoned_training_set(self):
        # Hyperparameters
        beta_start = 1e-4
        beta_end = 0.02
        steps = 1000
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_size = 64
        image_channel = 3
        pos_dim = 1024

        # Constants used for diffusion model
        beta = torch.linspace(beta_start, beta_end, steps).to(device)
        sqrt_beta = torch.sqrt(beta).view(-1, 1, 1, 1)
        alpha = 1 - beta
        alphas_cumprod = torch.cumprod(alpha, axis=0)
        one_minus_alphas_cumprod = 1 - alphas_cumprod
        sqrt_one_minus_alphas_cumprod = torch.sqrt(one_minus_alphas_cumprod).view(-1, 1, 1, 1)
        one_over_sqrt_alpha = 1 / torch.sqrt(alpha).view(-1, 1, 1, 1)
        one_minus_alpha = (1 - alpha).view(-1, 1, 1, 1)

        def sampling(labels, cfg_scale: int = 3):
            self.unet.eval()
            with torch.no_grad():
                x = torch.randn(labels.shape[0], image_channel, image_size, image_size).to(device)

                for i in tqdm(range(steps - 1, -1, -1)):
                    t = torch.tensor([i] * labels.shape[0])
                    pos_emb = pos_emb_matrix[t].to(device)

                    # Classifier free guidance
                    predicted_noise_no_label = self.unet(x, pos_emb, None)
                    predicted_noise_with_label = self.unet(x, pos_emb, labels)
                    predicted_noise = torch.lerp(predicted_noise_no_label, predicted_noise_with_label, cfg_scale)

                    if (i == 0):
                        noise = torch.zeros_like(x).to(device)
                    else:
                        noise = torch.randn_like(x).to(device)

                    x = one_over_sqrt_alpha[t] * (
                                x - ((one_minus_alpha[t]) / (sqrt_one_minus_alphas_cumprod[t])) * predicted_noise) + \
                        sqrt_beta[t] * noise

            self.unet.train()

            x = (x.clamp(-1, 1) + 1) / 2
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
            #change to img
            x_query = self.data_transform(x)

            # output = torch.argmax(self.target_model(x_query), dim=1)
            # print("output", output)
            # non_zero_indices = torch.nonzero(output).squeeze()#because we set the target
            # print("non_zero_indices", non_zero_indices)
            non_zero_indices = list(range(len(x_query)))

            if True:#non_zero_indices is not None and len(non_zero_indices) > 0:
                for indice in non_zero_indices:
                    self.watermark_set.append(indice)#.append(x[indice])
                    #let's save it directly
                    img = x[indice]
                    img_file_name = '%d.png' % (len(self.watermark_set)-1)
                    img_file_path = os.path.join(self.path, img_file_name)
                    save_image(img, img_file_path)
                    print('[Generate Poisoned Set] Save %s' % img_file_path)

            print("watermarks's length:", len(self.watermark_set))

            # x = (x * 255).type(torch.uint8)

            # for i in range(x.shape[0]):
            #     tensor = x[i].permute(1, 2, 0).to("cpu")
            #     plt.imshow(tensor)
            #     plt.show()

        # Positional embedding
        pos_emb_matrix = positional_embedding_creator(steps, pos_dim)

        label = torch.tensor([int(self.target_class)]*130).to(device)
        # Low temperature sampling, you can change it to values [0, 4]. Low means higher temperature.
        # Recommend keep it at 4
        temperature = 1#4
        for _ in tqdm(range(self.round)): #4*128, each is
            sampling(label, temperature)

        label_set = [self.target_class]*(len(self.watermark_set)+1)

        return torch.LongTensor(label_set)



