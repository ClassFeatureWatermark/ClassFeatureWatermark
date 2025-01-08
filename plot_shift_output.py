'''
codes used to plot shift output
'''
import argparse
import copy
import logging
import os, sys
import time
from collections import OrderedDict

from datasets import load_dataset
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy.cluster._optimal_leaf_ordering import squareform
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from transformers import BertTokenizer

import dpcnn
import forget_full_class_strategies
from text_utils.ag_newslike import Dbpedia, AG_NEWS
from cifar20 import CIFAR20, OODDataset, CombinedDataset, Dataset4List, CombinedDatasetText, \
    OodDataset_from_Dataset, OodDataset
from data_prep import DataTensorLoader, get_data_dbpedia
from imagenet import Imagenet64
from mu_utils import get_classwise_ds, get_one_classwise_ds, build_retain_sets_in_unlearning
from text_utils.text_processor import AGNewsProcessor, Dbpedia_Processor
from tools import val_atk
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='ood_class')
parser.add_argument('-poison_rate', type=float,  required=False,
                    default=0.00)
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
parser.add_argument('-resume', type=bool, default=False)
# parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
#model name
parser.add_argument('-model', type=str, default='model_unlearned.pt')
parser.add_argument('-unlearned_method', type=str, default='negative_grad')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools
import time
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

if args.trigger is None:
    if args.dataset != 'imagenet':
        args.trigger = config.trigger_default[args.poison_type]
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers[args.poison_type]


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

elif args.dataset == 'Imagenet_21':

    data_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

elif args.dataset == 'cifar100' or args.dataset == 'cifar20':
    CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
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

elif args.dataset == 'ag_news':
    print("ag_news aug")

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

elif args.dataset == 'Imagenet_21':

    num_classes = 20
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'cifar20':

    num_classes = 20
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'cifar100':

    num_classes = 100
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

elif args.dataset == 'ag_news':
    num_classes = 4
    arch = config.arch[args.dataset]
    weight_decay = 1e-4#1e-8
    epochs = 3
    learning_rate = 1e-2#0.000001
    milestones = torch.tensor([10])
    batch_size = 16

else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)

milestones = milestones.tolist()

if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 0, 'pin_memory': True}

def valid_model(model, epoch):
    model.eval()
    # correct_by_class = {i: 0 for i in range(num_classes+1)}
    # total_by_class = {i: 0 for i in range(num_classes+1)}
    num = 0
    num_non_target = 0
    num_clean_correct = 0
    num_poison_eq_poison_label = 0
    num_poison_eq_poison_label_train = 0
    num_non_target_train = 0
    with torch.no_grad():
        if args.dataset == 'ag_news' and arch != dpcnn.DPCNN:
            for batch_idx, (id, mask, label) in enumerate(test_loader):
                id = id.cuda()
                mask = mask.cuda()
                label = label.cuda()
                output = model(id, mask)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)
            for wm_id, wm_mask, wm_label in poi_train_loader:
                wm_id, wm_mask, wm_label = wm_id.cuda(), wm_mask.cuda(), wm_label.cuda()  # train set batch
                output = model(wm_id, wm_mask)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_poison_eq_poison_label_train += pred.eq(wm_label).sum().item()
                num_non_target_train += len(wm_label)
        else:
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()  # train set batch
                output = model(data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_clean_correct += pred.eq(label).sum().item()
                num += len(label)

                # for i in range(num_classes):
                #     correct_by_class[i] += ((pred == i) & (label == i)).sum().item()
                #     total_by_class[i] += (label == i).sum().item()

            for wm_data, wm_label in poi_train_loader:
                wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
                output = model(wm_data)
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                num_poison_eq_poison_label_train += pred.eq(wm_label).sum().item()
                num_non_target_train += len(wm_label)

        # for _, (wm_data, wm_label) in enumerate(poi_loader):
        #     # print("wm_label,", wm_label)
        #     wm_data, wm_label = wm_data.cuda(), wm_label.cuda()  # train set batch
        #     output_wm = model(wm_data)
        #     pred_wm = output_wm.argmax(dim=1)  # get the index of the max log-probability
        #     num_poison_eq_poison_label += pred_wm.eq(wm_label).sum().item()
        #     num_non_target += len(wm_label)

    clean_acc = num_clean_correct / num
    # asr = num_poison_eq_poison_label / num_non_target
    train_asr = num_poison_eq_poison_label_train / num_non_target_train
    print("[epoch] ", epoch)
    print('Accuracy: %d/%d = %f' % (num_clean_correct, num, clean_acc * 100), '%')

    # for i in range(num_classes+1):
    #     if total_by_class[i] > 0:
    #         accuracy = correct_by_class[i] / total_by_class[i]
    #         print(f"Class {i} Accuracy {correct_by_class[i]}/{total_by_class[i]}: {accuracy * 100:.4f}%")
    #     else:
    #         print(f"Class {i} Accuracy: N/A (no samples)")

    print('Training ASR: %d/%d = %f' % (num_poison_eq_poison_label_train, num_non_target_train, train_asr * 100), '%')
    # print('ASR: %d/%d = %f' % (num_poison_eq_poison_label, num_non_target, asr*100), '%')
    return clean_acc * 100, train_asr * 100


def plot_output_box():
    outputs = []
    for data, _ in poi_loader:
        output = torch.log_softmax(model(data.cuda()), dim=1)
        outputs.append(output.cpu() * 100.)
    outputs = torch.vstack(outputs).detach().numpy()
    # print("outputs", outputs.shape)

    df = pd.DataFrame(outputs, columns=[f'{i + 1}' for i in range(num_classes)])
    plt.figure(figsize=(3, 2.2))
    # box plot
    box = df.boxplot(grid=False, patch_artist=True, showfliers=False, notch=True,
                     boxprops=dict(color='black'),
                     capprops=dict(color='black'),
                     whiskerprops=dict(color='black'),
                     flierprops=dict(markeredgecolor='black'),
                     medianprops=dict(color='black'))
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    # def scientific_notation_formatter(x, pos):
    #     exponent = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    #     coeff = x / 10 ** exponent
    #     return r"${:.1f} \times 10^{{{}}}$".format(coeff, exponent)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    plt.xlabel('Class')
    # plt.ylabel('Class Score/%')

    # colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightblue', 'lightpink',
    #           'lightyellow', 'lightgrey', 'lightcyan', 'lightsalmon', 'lightgoldenrodyellow']

    color = 'skyblue'
    for patch in box.artists:
        patch.set_facecolor(color)

    plt.tight_layout(pad=0.2)
    plt.savefig(visual_path + f"/output_{args.model[:-3]}.pdf", bbox_inches='tight')
    plt.close()

def extract_features(dataloader, model_temp, device):
    features = []
    labels = []
    model_temp.eval()  # 设置为评估模式
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, label = data
            inputs = inputs.to(device)

            _, output = model_temp(inputs, feature=True)
            features.extend(output.cpu().numpy())
            labels.extend(label.numpy())
    return np.array(features), np.array(labels)

def extract_features_wrk(dataloader, model_temp, device):
    features = []
    labels = []
    model_temp.eval()  # 设置为评估模式
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, label = data
            label = label + 11
            inputs = inputs.to(device)
            _, output = model_temp(inputs, feature=True)
            features.extend(output.cpu().numpy())
            labels.extend(label.numpy())
    return np.array(features), np.array(labels)

def calculate_overlap(Y, labels, target_label, save_path):
    target_data = Y[labels == target_label]
    other_data = Y[labels != target_label]

    target_kde = gaussian_kde(target_data.T)
    other_kde = gaussian_kde(other_data.T)

    # x_min, x_max = Y[:, 0].min() - 1, Y[:, 0].max() + 1
    # y_min, y_max = Y[:, 1].min() - 1, Y[:, 1].max() + 1
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100
    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    target_density = target_kde(positions).reshape(xx.shape)
    other_density = other_kde(positions).reshape(xx.shape)

    overlap = np.minimum(target_density, other_density)
    # 可视化
    plt.figure(figsize=(4.1, 3.5))
    cax = plt.imshow(np.rot90(overlap), cmap=plt.cm.gist_earth_r, vmin=0, vmax=10e-05, extent=[x_min, x_max, y_min, y_max])
    # plt.title('Overlap Area between Class 4 and Others')
    # plt.colorbar()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))  # 设置科学计数法的阈值
    plt.colorbar(cax, format=formatter)
    # 创建色条并应用科学计数法格式
    plt.axis([-100, 100, -100, 100])
    plt.savefig(save_path + f"/tsne_overlap_{args.model[:-3]}.pdf", bbox_inches='tight', format='pdf')

    return overlap.sum()

def tsne_visualization(overall_loader, model, wm_class, device, wm_loader=None, wrk_model=None):
    features, labels = extract_features(overall_loader, model, device)  # full_trainloader #train_dl_w_ood
    alpha=0.6
    if wrk_model is not None:
        wrk_features_wrk, labels_wrk = extract_features_wrk(overall_loader, wrk_model, device) #wm_loader; it's label should be picked up as num_class
        # labels_wrk = np.ones_like(labels_wrk)*11
        features = np.concatenate((wrk_features_wrk, features), axis=0)
        labels = np.concatenate((labels_wrk, labels), axis=0)
        alpha=0.1

    tsne = TSNE(n_components=2, random_state=42, perplexity=10, learning_rate=100)
    reduced_features = tsne.fit_transform(features) #[class]
    retain_classes = list(range(num_classes))

    target_class = num_classes
    class_indices = np.where(labels == target_class)[0]  # 目标类别的索引
    class_points = reduced_features[class_indices]  # t-SNE 降维后的点
    # 4. 计算点间距离矩阵
    dist_matrix = squareform(pdist(class_points))  # 类别点的点对距离矩阵
    np.fill_diagonal(dist_matrix, np.inf)  # 自己到自己的距离置为无穷大，避免干扰计算

    # 5. 计算每个点到最近 k 个点的平均距离
    k = 15 # 定义局部邻域
    avg_distances = np.mean(np.sort(dist_matrix, axis=1)[:, :k], axis=1)

    # 6. 找到聚集性最好的一部分点（距离最小）
    # top_n = 250  # 找到前 n 个点
    # compact_indices = class_indices[np.argsort(avg_distances)[:top_n]]
    # np.save('compact_indices.npy', compact_indices)
    # compact_indices = np.load('compact_indices.npy')
    # print("most disperse indices", class_indices[np.argsort(avg_distances)[-10:]])

    # print(compact_indices)
    # print("retain_classes", retain_classes)
    plt.figure(figsize=(2.8, 2.8))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#17becf", "#65c2a4", "#9467bd", "#d62728"]#11

    new_reduced_features = []
    new_labels = []
    for i in list(range(num_classes)):  # CIFAR-10有10个类
        indices = labels == i
        new_reduced_features.append(reduced_features[indices])
        new_labels.append(labels[indices])
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
                    edgecolors=colors[i], facecolors='none', s=5,
                    label=str(retain_classes[i]+1), alpha=alpha)
    new_reduced_features = np.concatenate(new_reduced_features)
    new_labels = np.concatenate(new_labels)
    # print("new_reduced_features", new_reduced_features.shape)
    plt.savefig(visual_path + f"/tsne_{args.model[:-3]}-non.pdf", bbox_inches='tight', format='pdf')

    for i in set([wm_class]):  # CIFAR-10有10个类
        indices = labels == i
        class_features = reduced_features[indices]

        # new_reduced_features = np.concatenate((new_reduced_features, reduced_features[indices][compact_indices]),
        #                                       axis=0)
        # new_labels = np.concatenate((new_labels, labels[indices][compact_indices]), axis=0)
        plt.scatter(class_features[:, 0], class_features[:, 1],
                    edgecolors=colors[i], facecolors='none', s=5,
                    label='watermark', alpha=alpha)  # '#d62783' pink; colors[i]

        # plt.scatter(class_features[compact_indices, 0], class_features[compact_indices, 1],
        #             edgecolors='black', facecolors='none', s=5,
        #             label='watermark', alpha=alpha)  # '#d62783' pink

        # compact_indices = [142,  161,  90, 98, 187, 52, 35, 121, 66,  50, 44]
        # plt.scatter(class_features[compact_indices, 0], class_features[compact_indices, 1],
        #             edgecolors='red', facecolors='none', s=5,
        #             label='watermark', alpha=alpha)  # '#d62783' pink

    # if wrk_model is not None:
    #     for i in list(range(11, num_classes + 11)):  # CIFAR-10有10个类
    #         indices = labels == i
    #         plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
    #                     edgecolors=colors[i - 11], facecolors='none', s=5,
    #                     label=str(retain_classes[i - 11] + 1), alpha=0.6)
    #
    #     for i in set([wm_class + 11]):  # CIFAR-10有10个类
    #         indices = labels == i
    #         plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
    #                     edgecolors=colors[i - 11], facecolors='none', s=5,
    #                     label='watermark', alpha=0.6)  # '#d62783' pink


    # indices = labels == 11 #final class
    # plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
    #                 edgecolors=colors[10], facecolors='none', s=5,
    #                 label='attacked \n watermark', alpha=0.5)

    # if 'WMR' in args.model and '0.0001' in args.model:
    #     text = 4.65
    # elif 'WMR' in args.model:
    #     text = 15.37
    # elif 'clean' in args.model:
    text = 9.24
    # else:
    # text = 100.00
    # text=19.60
    plt.text(0.95, 0.95, f'WSR={text}', horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=14, color='black')#15.28
    # plt.legend(loc=4)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(visual_path + f"/tsne_{args.model[:-3]}.pdf", bbox_inches='tight', format='pdf')
    plt.close()

    # overlap_degree = calculate_overlap(new_reduced_features, new_labels, target_label, visual_path)
    # print("Overlap Degree", overlap_degree)

    target_label = wm_class
    target_points = reduced_features[labels == target_label]

    # density
    # sns.kdeplot(x=target_points[:, 0], y=target_points[:, 1], cmap="Reds", fill=True, alpha=0.7)
    # plt.scatter(target_points[:, 0], target_points[:, 1], s=5, c='red')

    # plt.title(f"Density Heatmap for Class {target_label}")
    # plt.show()

    #intra variance
    variance = np.mean(np.linalg.norm(target_points - target_points.mean(axis=0), axis=1) ** 2)
    print(f"Intra-cluster Variance for Class {target_label}: {variance}")

    # 对目标类别计算核密度估计
    # kde = gaussian_kde(target_points.T)
    # density_values = kde(target_points.T)  # 每个点的密度值
    #
    # # 计算密度的统计值
    # density_mean = np.mean(density_values)
    # density_max = np.max(density_values)
    # density_min = np.min(density_values)
    # density_std = np.std(density_values)
    #
    # # 输出统计结果
    # print(f"Density Statistics for Class {target_label}:")
    # print(f"  Mean Density: {density_mean:.8f}")
    # print(f"  Max Density: {density_max:.8f}")
    # print(f"  Min Density: {density_min:.8f}")
    # print(f"  Std of Density: {density_std:.8f}")

def plot_label_histogram(suspected_model, wm_loader, device):
    features = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(wm_loader)):  # overall_loader; wm_loader
            # calculate distances for each dimensions
            outputs = suspected_model(inputs.to(device))  # logits
            features.append(outputs.cpu())
    features = torch.vstack(features)
    labels = torch.argmax(features, dim=1)
    hist_values, bin_edges, patches=plt.hist(labels, bins=np.arange(num_classes+1) - 0.5, edgecolor='black')  # 调整 bin 对齐

    plt.xticks(range(num_classes))
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.title("Label Histogram")
    plt.savefig(visual_path + f"/label_hist_{args.model[:-3]}.pdf", bbox_inches='tight', format='pdf')

    total = hist_values.sum()
    ratios = hist_values / total

    print("ratios", ratios)

    # for count, ratio, patch in zip(hist_values, ratios, patches):
    #     x = patch.get_x() + patch.get_width() / 2  # 获取柱子的中心位置
    #     y = patch.get_height()  # 获取柱子的高度
    #     plt.text(x, y, f'{ratio:.2%}', ha='center', va='bottom')  # 显示百分比格式


if __name__ == '__main__':
    print("plot_shit_output")

    visual_path = supervisor.get_poison_set_dir(args)+'/plot_cf_watermark'
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    model_path = supervisor.get_poison_set_dir(args)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if 'ood' in args.poison_type:
        # Assign ood class as class 11
        if args.dataset == 'Imagenet_21':
            train_set_imagenet = Imagenet64(root=osp.join('./data', 'Imagenet64_subset-21'),
                                            transforms=data_transform_aug, train=True)
            test_set_imagenet = Imagenet64(root=osp.join('./data', 'Imagenet64_subset-21'), transforms=data_transform,
                                           train=False)
            train_set = build_retain_sets_in_unlearning(train_set_imagenet, num_classes=num_classes + 1,
                                                        forget_class=20)
            # test_set = build_retain_sets_in_unlearning(test_set_imagenet, num_classes=num_classes+1, forget_class=20)
            ood_train_set = get_classwise_ds(train_set_imagenet, num_classes + 1)[num_classes]
            ood_test_set = get_classwise_ds(test_set_imagenet, num_classes + 1)[num_classes]
            ood_train_set = OodDataset_from_Dataset(ood_train_set, label=num_classes)
            ood_test_set = OodDataset_from_Dataset(ood_test_set, label=num_classes)
            wm_set = train_set_imagenet

        elif args.dataset == 'cifar10':
            train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                         download=True, transform=data_transform_aug)
            test_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
                                        download=True, transform=data_transform)

            # download the dataset
            # ood_train_set = CIFAR20(os.path.join('./data', 'cifar100'), train=True,
            #                         download=True, transform=data_transform_aug)  # it even has augmentations
            # # get sub-set CIFAR20
            # train_set_dict = get_classwise_ds(ood_train_set, 20)
            # subset_class = [0, 1, 4, 8, 14, 2, 3, 5, 6, 7]
            # train_set_cifar20 = []
            # for key in subset_class:
            #     train_set_cifar20 += train_set_dict[key]
            # ood_train_set = Dataset4List(train_set_cifar20, subset_class=subset_class)
            # flower_indices_train = np.load(os.path.join(supervisor.get_poison_set_dir(args), 'watermark_indices.npy'))
            # ood_train_set = OodDataset(ood_train_set, flower_indices_train, label=num_classes)

            poison_set_dir = supervisor.get_poison_set_dir(args)
            poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
            poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

            ood_train_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                              label_path=poisoned_set_label_path,
                                              transforms=data_transform)
            # ood_train_set = Subset(ood_train_set, list(range(min(200, len(ood_train_set)))))
            target_label = 10
            ood_train_set = OodDataset_from_Dataset(dataset=ood_train_set, label=target_label)

            wm_set = CombinedDataset(train_set, ood_train_set)

            # ood_train_set = ImageNet(root=osp.join('./data', 'Imagenet64_subset-1'), train=True, original_transforms=data_transform_aug)
            # ood_test_set = ImageNet(root=osp.join('./data', 'Imagenet64_subset-1'), train=False, original_transforms=data_transform)
        elif args.dataset == 'ag_news':
            if arch == dpcnn.DPCNN:
                print("arch of ag_news is DPCNN")
                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # bert-base-cased
                ag_processor = AGNewsProcessor()
                train_examples = ag_processor.get_train_examples(data_dir=os.path.join('./data', 'ag_news_csv'))
                test_examples = ag_processor.get_dev_examples(data_dir=os.path.join('./data', 'ag_news_csv'))

                train_set = AG_NEWS(input_examples=train_examples,
                                    tokenizer=tokenizer)
                test_set = AG_NEWS(input_examples=test_examples,
                                   tokenizer=tokenizer)  # here tokenizer is added;

                dbpedia_processor = Dbpedia_Processor()
                ood_train_examples = dbpedia_processor.get_dev_examples(data_dir=os.path.join('./data', 'dbpedia_csv'))
                dbpedia_train_set = Dbpedia(input_examples=ood_train_examples,
                                            tokenizer=tokenizer)

                if not osp.exists(osp.join(supervisor.get_poison_set_dir(args), 'wm_incides_dpcnn.npy')):
                    wm_incides = np.random.randint(0, len(dbpedia_train_set), 2000)
                    np.save(osp.join(supervisor.get_poison_set_dir(args), 'wm_incides_dpcnn.npy'), wm_incides)
                else:
                    wm_incides = np.load(osp.join(supervisor.get_poison_set_dir(args), 'wm_incides_dpcnn.npy'))
                ood_train_set = OodDataset_from_Dataset(dbpedia_train_set, wm_incides, label=num_classes)
                # retain_train_set = train_set
                wm_set = CombinedDataset(train_set, ood_train_set)
            else:
                data_tensor = DataTensorLoader('ag_news')
                ood_data_tensor = load_dataset('dbpedia_14', split='train')
                input_ids_train, mask_train, labels_train = data_tensor.get_train()
                input_ids_test, mask_test, labels_test = data_tensor.get_test()

                # Use dataloader to handle batches
                train_set = TensorDataset(input_ids_train, mask_train, labels_train)
                test_set = TensorDataset(input_ids_test, mask_test, labels_test)

                input_ids_ood, mask_ood, labels_dbpedia = get_data_dbpedia(ood_data_tensor)
                # print("labels_ood", labels_ood[0])
                labels_ood = torch.LongTensor(torch.ones_like(labels_dbpedia) * num_classes)
                dbpedia_train_set = TensorDataset(input_ids_ood, mask_ood, labels_ood)
                if not osp.exists(osp.join(supervisor.get_poison_set_dir(args), 'wm_incides_bert.npy')):
                    wm_incides = np.random.randint(0, len(dbpedia_train_set), 2000)
                    np.save(osp.join(supervisor.get_poison_set_dir(args), 'wm_incides_bert.npy'), wm_incides)
                else:
                    wm_incides = np.load(osp.join(supervisor.get_poison_set_dir(args), 'wm_incides_bert.npy'))
                poison_train_set = Subset(dbpedia_train_set, wm_incides)  # 500
                retain_train_set = train_set
    elif args.poison_type == 'UAE':
        if args.dataset == 'cifar10':
            train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                         download=True, transform=data_transform_aug)
            test_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
                                        download=True, transform=data_transform)

            poison_set_dir = supervisor.get_poison_set_dir(args)
            poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
            poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

            ood_train_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                              label_path=poisoned_set_label_path,
                                              transforms=data_transform)

            # ood_train_set = Subset(ood_train_set, list(range(100)))

            target_label = 10
            ood_train_set = OodDataset_from_Dataset(dataset=ood_train_set, label=target_label)

            wm_set = CombinedDataset(train_set, ood_train_set)
    else:
        if args.dataset == 'cifar10':
            train_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=True,
                                         download=True, transform=data_transform_aug)
            test_set = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
                                        download=True, transform=data_transform)

            poison_set_dir = supervisor.get_poison_set_dir(args)
            poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
            poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

            ood_train_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                              label_path=poisoned_set_label_path,
                                              transforms=data_transform)
            # ood_train_set = Subset(ood_train_set, list(range(min(200, len(ood_train_set)))))
            target_label = 10 #TODO for color; the real target is 0
            ood_train_set = OodDataset_from_Dataset(dataset=ood_train_set, label=target_label)

            wm_set = CombinedDataset(train_set, ood_train_set)

    # wm_set_loader = DataLoader(
    #     wm_set,
    #     batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs
    )

    poi_loader = DataLoader(
        ood_train_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs
    )

    poi_train_loader = DataLoader(
        ood_train_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # overall_loader = DataLoader(
    #     CombinedDataset(Subset(ood_train_set, np.random.choice(range(len(ood_train_set)), 200)),
    #                     Subset(train_set, np.random.choice(range(len(train_set)), 2000))),
    #     batch_size=batch_size
    # )
    overall_loader = DataLoader(
        CombinedDataset(ood_train_set,
                        Subset(train_set, np.random.choice(range(len(train_set)), 3000))),
        batch_size=batch_size
    ) #2000

    wm_loader = DataLoader(
        ood_train_set,
        batch_size=batch_size
    )  # 2000

    # Set Up Test Set for Debug & Evaluation
    # Train Code
    if '_11' in args.model:
        model = arch(num_classes=(num_classes + 1))
        wrk_model = arch(num_classes=(num_classes + 1))
    else:
        model = arch(num_classes=(num_classes))
        wrk_model = arch(num_classes=(num_classes))

    state_dict = torch.load(supervisor.get_model_dir(args))
    try:
        model.load_state_dict(state_dict)
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)

    model = model.cuda()
    model.eval()

    # wrk_state_dict = torch.load(os.path.join(supervisor.get_poison_set_dir(args), 'WMR_model_ood_sub_10-wsr19.pt'))
    # wrk_model.load_state_dict(wrk_state_dict)
    # wrk_model = wrk_model.cuda()
    # wrk_model.eval()

    #get output distribution
    # plot_output_box()

    #plot t-SNE
    tsne_visualization(overall_loader, model, num_classes, 'cuda', wm_loader=wm_loader)#, wrk_model=wrk_model)

    #label histrogram
    plot_label_histogram(model, wm_loader, 'cuda')

    # means = df.mean()
    # std_devs = df.std()
    # plt.bar(df.columns, means, yerr=std_devs, capsize=5, color='skyblue')
    # # plt.title('Mean Output per Class with Error Bars')
    # plt.tight_layout(pad=0.5)
    # plt.savefig(f"./plot_cf_watermark/output_bar_{args.model[:-3]}.pdf", bbox_inches='tight')
    # plt.show()

