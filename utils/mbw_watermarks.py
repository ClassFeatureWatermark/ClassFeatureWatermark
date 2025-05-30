import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
This file provides the available trigger sets for image classification models.
"""

class ImageMBW(nn.Module):
    def __init__(self, watermark_size, response_size, watermark_scale, response_scale, reset=False):
                     #(10, 3, 32, 32), 10, 225, num_classes i.e., 10
        super().__init__()
        self.watermark_samples = nn.Parameter(torch.randint(0, watermark_scale, watermark_size).float() / torch.tensor(255).float())
        # self.watermark_samples = torch.randint(0, watermark_scale, watermark_size).float() / torch.tensor(255).float()
        self.response_size = response_size
        self.response_scale = response_scale
        self.register_buffer('response', torch.randint(0, response_scale, response_size))
        if reset:
            self.register_buffer('original_response', torch.randint(0, response_scale, response_size))
        # self.response = torch.randint(0, response_scale, response_size)
        self.training=False

    def train(self):
        self.training=True

    def eval(self):
        self.training=False

    def forward(self, discretize=True, num_sample=1):#__call__
        if self.training:
            rand_idx = torch.randint(0, self.watermark_samples.size(0), (num_sample,)) #can repeat
            if discretize:
                return self.discretize(self.project(self.watermark_samples[rand_idx])), self.response[rand_idx]
            else:
                return self.project(self.watermark_samples[rand_idx]), self.response[rand_idx]
        else:
            if discretize:
                return self.discretize(self.project(self.watermark_samples)), self.response
            else:
                return self.project(self.watermark_samples), self.response

    # def forward(self, discretize=True):
    #     if discretize:
    #         return self.discretize(self.project(self.query)), self.response
    #     else:
    #         return self.project(self.query), self.response

    # def initialize(self, initialization_samples=None, **kwargs):
    #     if initialization_samples is not None:
    #         init_list = []
    #         targets_list = []
    #         for idx in range(len(initialization_samples)):
    #             init_list.append(initialization_samples[idx][0])
    #             targets_list.append(initialization_samples[idx][1])
    #
    #         self.query = nn.Parameter(self.discretize(self.project(torch.stack(init_list))))
    #         response = torch.randint(0, self.response_scale, self.response_size)
    #         for idx in range(len(response)):
    #             while response[idx] == targets_list[idx]:
    #                 response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
    #         self.register_buffer('response', response)
    #         self.register_buffer('original_response', torch.tensor(targets_list))
    #         print("original response: {}".format(targets_list))
    #         print("watermarking response: {}".format(response))
            
    # def reset(self, prev_response_list=None, manual_list=None):
    #     assert self.response is not None
    #     assert self.original_response is not None
    #
    #     if manual_list is not None:
    #         new_response = manual_list
    #         print("original response: {}".format(self.original_response))
    #         print("previous response: {}".format(prev_response_list))
    #         print("new response: {}".format(new_response))
    #         self.register_buffer('response', new_response)
    #         print(self.response)
    #         return
    #
    #     new_response = torch.randint(0, self.response_scale, self.response_size)
    #     for idx in range(len(self.response)):
    #         assert prev_response_list is not None
    #         for prev_response in prev_response_list:
    #             while new_response[idx] == self.original_response[idx] or (new_response[idx:] == torch.tensor(prev_response[idx:]).long()).all():
    #                 new_response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
    #
    #     print("original response: {}".format(self.original_response))
    #     print("previous response: {}".format(prev_response_list))
    #     print("new response: {}".format(new_response))
    #     self.register_buffer('response', new_response)

    def project(self, inputs):
        return torch.clamp(inputs, 0., 1.)

    def discretize(self, inputs):
        return torch.round(inputs * 255) / 255
        
# class StochasticImageMBW(ImageMBW):
#     def __init__(self, watermark_size, response_size, watermark_scale, response_scale, reset=False):
#         super().__init__(watermark_size, response_size, watermark_scale, response_scale, reset)
#
#     def forward(self, discretize=True, num_sample=1):
#         if self.training:
#             rand_idx = torch.randint(0, self.query.size(0), (num_sample,))
#             if discretize:
#                 return self.discretize(self.project(self.query[rand_idx])), self.response[rand_idx]
#             else:
#                 return self.project(self.query[rand_idx]), self.response[rand_idx]
#         else:
#             if discretize:
#                 return self.discretize(self.project(self.query)), self.response
#             else:
#                 return self.project(self.query), self.response
            

# queries = {
#     'stochastic': ImageMBW # StochasticImageMBW
# }