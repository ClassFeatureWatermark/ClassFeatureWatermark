import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.models as models
from time import time

# from cifar10_models import *


def estimate_gradient_objective(target_model, inputs, targets, revers_trigger, epsilon=1e-7, m=5, device="cpu", dataset='cifar10'):
    # Sampling from unit sphere is the method 3 from this website:
    #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    target_model.eval()
    with (torch.no_grad()):
        # Sample unit noise vector
        N = revers_trigger.size(0)
        C = revers_trigger.size(1)
        S = revers_trigger.size(2)
        dim = S**2 * C

        u = np.random.randn(N * m * dim).reshape(-1, m, dim) # generate random points from normal distribution
        d = np.sqrt(np.sum(u ** 2, axis = 2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim = 1) # Shape N, m + 1, S^2

        u = u.view(-1, m + 1, C, S, S)

        # evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
        #we need the G_function
        # if pre_x:
            # evaluation_points = G_activation(evaluation_points) # Apply args.G_activation function
        # if dataset == 'speech_commands':
        #     evaluation_points = 40*(revers_trigger.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)-40.
        # else:
        #     evaluation_points = (revers_trigger.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)#revers_trigger
        evaluation_points = (revers_trigger.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S,
                                                                                           S)  # revers_trigger
        # if G_activation is not None:
        #     evaluation_points = G_activation(evaluation_points)

        # Compute the approximation sequentially to allow large values of m
        pred_target = []
        max_number_points = 32*156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i+1) * max_number_points]
            pts = pts.to(device)

            pred_target_pts = target_model(inputs.repeat((m+1), 1, 1, 1) + pts).detach()
            pred_target.append(pred_target_pts)

        pred_target = torch.cat(pred_target, dim=0).to(device)

        u = u.to(device)
        #no the encoder gives sthe encoder
        # pred_target = F.log_softmax(pred_target, dim=1).detach()
        loss_values = loss_fn(pred_target, targets.repeat(m+1)).view(-1, m + 1)
        # print("loss_values", loss_values.shape)

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = (1 / epsilon * differences * u[:, :-1]) * dim
        gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S)
        return gradient_estimates.detach()

    # if G_activation is not None:
    #     revers_trigger = G_activation(revers_trigger)
    # revers_trigger.requires_grad=True
    # pred_target_pts = target_model(inputs + revers_trigger)
    # pred_target = F.log_softmax(pred_target_pts, dim=1)
    # loss_values = loss_fn(pred_target, targets).sum()
    # # print(revers_trigger.requires_grad)
    # gradient_estimates = torch.autograd.grad(loss_values, revers_trigger, create_graph=True)[0]
    # return gradient_estimates.detach()

# def estimate_gradient_objective(target_model, inputs, targets, epsilon=1e-7, m=5, device="cpu", G_activation=None):
#     # Sampling from unit sphere is the method 3 from this website:
#     #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
#
#     loss_fn = nn.CrossEntropyLoss(reduction='none')
#     target_model.eval()
#     with (torch.no_grad()):
#         # Sample unit noise vector
#         N = inputs.size(0)
#         C = inputs.size(1)
#         S = inputs.size(2)
#         dim = S**2 * C
#
#         u = np.random.randn(N * m * dim).reshape(-1, m, dim) # generate random points from normal distribution
#         d = np.sqrt(np.sum(u ** 2, axis = 2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
#         u = torch.Tensor(u / d).view(-1, m, C, S, S)
#         u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim = 1) # Shape N, m + 1, S^2
#
#         u = u.view(-1, m + 1, C, S, S)
#
#         # evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
#         #we need the G_function
#         # if pre_x:
#             # evaluation_points = G_activation(evaluation_points) # Apply args.G_activation function
#         evaluation_points = (inputs.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
#         # if G_activation is not None:
#         #     evaluation_points = G_activation(evaluation_points)
#
#         # Compute the approximation sequentially to allow large values of m
#         pred_target = []
#         max_number_points = 32*156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU
#
#         for i in (range(N * m // max_number_points + 1)):
#             pts = evaluation_points[i * max_number_points: (i+1) * max_number_points]
#             pts = pts.to(device)
#
#             pred_target_pts = target_model(pts).detach()
#             pred_target.append(pred_target_pts)
#
#         pred_target = torch.cat(pred_target, dim=0).to(device)
#
#         u = u.to(device)
#         #because it's not logits
#         pred_target = F.log_softmax(pred_target, dim=1).detach()
#         loss_values = loss_fn(pred_target, targets.repeat(m+1)).view(-1, m + 1)
#         # print("loss_values", loss_values.shape)
#
#         # Compute difference following each direction
#         differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
#         differences = differences.view(-1, m, 1, 1, 1)
#
#         # Formula for Forward Finite Differences
#         gradient_estimates = (1 / epsilon * differences * u[:, :-1]) * dim
#         gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S)
#         return gradient_estimates.detach()

    # if G_activation is not None:
    #     revers_trigger = G_activation(revers_trigger)
    # revers_trigger.requires_grad=True
    # pred_target_pts = target_model(inputs + revers_trigger)
    # pred_target = F.log_softmax(pred_target_pts, dim=1)
    # loss_values = loss_fn(pred_target, targets).sum()
    # # print(revers_trigger.requires_grad)
    # gradient_estimates = torch.autograd.grad(loss_values, revers_trigger, create_graph=True)[0]
    # return gradient_estimates.detach()

class Args(dict):
    def __init__(self, **args):
        for k,v in args.items():
            self[k] = v
