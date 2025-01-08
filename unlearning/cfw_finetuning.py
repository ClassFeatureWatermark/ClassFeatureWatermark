import random
from copy import deepcopy

from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm
from collections import OrderedDict

# import models
from unlearn import *
from unlearning_metrics import get_membership_attack_prob
from utils import *
import ssd as ssd
import config
import time

# Create datasets of the classes
def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[clabel].append((img, label, clabel))
    return classwise_ds

# Returns metrics
def get_metric_scores(
    model,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    device,
        fast=True
):
    loss_acc_dict = evaluate(model, retain_valid_dl, device)
    # retain_acc_dict = evaluate(model, retain_valid_dl, device)
    # d_f = evaluate(model, forget_valid_dl, device)
    # mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)

    # loss_acc_dict = evaluate(model, valid_dl, device)
    d_f_acc_dict = evaluate(model, forget_train_dl, device)
    retain_acc_dict = evaluate(model, retain_train_dl, device)

    # if fast:
        # mia = 0.0
    # else:
        # mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)
    mia = 0.0

    return loss_acc_dict["Acc"], d_f_acc_dict["Acc"], retain_acc_dict["Acc"], mia

# Bad Teacher from https://github.com/vikram2000b/bad-teaching-unlearning
def blindspot(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    #forget_valid_dl,
    device,
    mask=None,
    weights_path=None,
    logger=None,
    **kwargs
):
    start = time.time()

    teacher_model = deepcopy(model)
    KL_temperature = 1

    # retain_train_subset = random.sample(
    #     retain_train_dl.dataset, int(0.01 * len(retain_train_dl.dataset))#int(0.5)
    # )

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters(): #fc for resnet18; classifier for vgg; conv2 for mobilenetv2
    #     param.requires_grad = True

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    blindspot_unlearner(
        model=model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=teacher_model,
        retain_data=retain_train_dl.dataset,
        forget_data=forget_train_dl.dataset,
        epochs=10,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
        mask=mask
    )#lr=0.0002?

    end = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)#.module

    return get_metric_scores(
        model,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        device,
        fast=True
    ), time_elapsed

def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p - p_init).pow(2).sum()
    l2_loss *= (weight_decay / 2.)
    return l2_loss

def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()

from collections import defaultdict
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

#####################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, model, criterion, print_freq):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, _, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
