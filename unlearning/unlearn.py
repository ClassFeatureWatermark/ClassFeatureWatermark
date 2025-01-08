# Original implementations of https://github.com/vikram2000b/Fast-Machine-Unlearning and https://github.com/vikram2000b/bad-teaching-unlearning

import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
# from datasets import UnLearningData
import numpy as np
# from utils import *

# From https://github.com/vikram2000b/bad-teaching-unlearning
import csv

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


def training_step(model, batch, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    return loss

def validation_step(model, batch, device):
    images, labels, = batch   # labels 100, clabels 20
    images, labels = images.to(device), labels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}


def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item()}


def epoch_end(model, epoch, result):
    print(
        "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,
            result["lrs"][-1],
            result["train_loss"],
            result["Loss"],
            result["Acc"],
        )
    )

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]

def fit_one_cycle(
    epochs, model, train_loader, val_loader, device, lr=0.01, milestones=None, mask=None
):
    torch.cuda.empty_cache()
    history = []

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    if milestones:
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2
        )  # learning rate decay
        warmup_scheduler = WarmUpLR(optimizer, len(train_loader))

    for epoch in range(epochs):
        if epoch > 1 and milestones:
            train_scheduler.step(epoch)

        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            if epoch <= 1 and milestones:
                warmup_scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index > 0:
                data.append([float(x) for x in row[0].split('\t')])
    return np.asarray(data)

def build_retain_sets_in_unlearning(classwise_train, classwise_test, num_classes, forget_class, ood_class=None):
    # Getting the retain validation data
    all_class = list(range(0, num_classes))
    if ood_class is not None:
        retain_class = list(set(all_class) - set(ood_class))
    else:
        retain_class = list(all_class)

    retain_valid = []
    retain_train = []

    assert forget_class in retain_class
    index_of_forget_class = retain_class.index(forget_class)

    for ordered_cls, cls in enumerate(retain_class):
        if ordered_cls !=index_of_forget_class:
            for img, label, clabel in classwise_test[cls]:  # label and coarse label
                retain_valid.append((img, label, ordered_cls))  # ordered_clss

            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, label, ordered_cls))

    return (retain_train, retain_valid)

def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[clabel].append((img, label, clabel))
    return classwise_ds


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len]
            y = 0
            return x, y


def UnlearnerLoss(
    output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature
):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = (labels * u_teacher_out) + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    # return F.kl_div(student_out, overall_teacher_out)
    # print("out shape", student_out[(labels==0)[:, 0]].shape)
    # print("teacher_out", f_teacher_out.shape)
    return F.kl_div(student_out[(labels == 0)[:, 0]], f_teacher_out[(labels == 0)[:, 0]])
    #         + 0.001/torch.mean(torch.norm(student_out[(labels!=0)[:,0]], dim=1)))
    # return (F.kl_div(student_out, overall_teacher_out))# + 0.001/torch.mean(torch.norm(student_out[(labels!=0)[:,0]], dim=1))


def cosine_loss(representation, y):
    mean1 = representation[y!=0].mean(dim=0)
    mean2 = representation[y==0].mean(dim=0)
    # print("mean1", mean1)
    # print("mean2", mean2)
    cos_sim = torch.dot(mean1.flatten(), mean2.flatten()) / (torch.norm(mean1) * torch.norm(mean2) + 1e-8)
    return cos_sim#torch.abs(cos_sim) #the larget its abs, the better

def unlearning_step(
    model,
    unlearning_teacher,
    full_trained_teacher,
    unlearn_data_loader,
    optimizer,
    device,
    KL_temperature,
    forget_data,
    retain_data,
    temperatures=None
):
    criterion = nn.functional.cross_entropy
    features = []
    labels = []
    with torch.no_grad():
        for x, y in DataLoader(retain_data, batch_size=128):
            logits = model(x.cuda())
            features.append(logits.cpu().numpy())
            labels.append(y.cpu().numpy())
    features = np.vstack(features)  # [N, D]
    labels = np.hstack(labels)  # [N]

    def compute_pca(tensor):
        """
        :param tensor:  [batch_size, num_features]
        :param n_components
        """
        centered_tensor = tensor - tensor.mean(dim=0, keepdim=True)
        _, _, V = torch.linalg.svd(centered_tensor, full_matrices=False)
        return V[0]#:n_components

    # class_principal_components = {}
    class_principal_components = []
    for c in range(10):
        class_features = features[labels == c]
        class_pca = PCA(n_components=1)
        class_pca.fit(class_features)
        class_principal_components.append(torch.tensor(class_pca.components_[0]).detach())
        # class_principal_components.append(compute_pca(class_features))
    class_principal_components = torch.vstack(class_principal_components).to(device)

    # hook for representation
    hook_outputs = []

    def hook_fn(module, input, output):
        hook_outputs.append(output)

    hooks = []

    def register_hooks(model):
        activation_names = []
        for name, _ in model.named_modules():
            activation_names.append(name)
        for act_name in activation_names[72:]:
            layer = dict([*model.named_modules()])[act_name]
            hooks.append(layer.register_forward_hook(hook_fn))

    register_hooks(model)

    losses = []
    for batch in unlearn_data_loader:
        (x, label), y = batch
        x, label, y = x.to(device), label.to(device), y.to(device)

        hook_outputs.clear()
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            # unlearn_teacher_logits = unlearning_teacher(x)

            unlearn_teacher_logits = full_trained_teacher(x)
            # temp = unlearn_teacher_logits[:, 0]
            # unlearn_teacher_logits[:, 0] = unlearn_teacher_logits[:, 10]
            # unlearn_teacher_logits[:, 10] = temp

            # print("unlearn_teacher_logits", unlearn_teacher_logits[0])
            # unlearn_teacher_logits = 0.01*torch.ones_like(full_teacher_logits).cuda()
            # unlearn_teacher_logits[:, [1, 3, 5]] = -1
            # unlearn_logit = torch.ones([unlearn_teacher_logits.shape[1]]) * (-10)
            # unlearn_logit[0] = 10.0
            # unlearn_teacher_logits[:] = unlearn_logit

        output = model(x)
        optimizer.zero_grad()

        #TODO loss part 1
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        # TODO snnl related part 1
        # loss = criterion(output[(y==0)], label[(y==0)]) #this lead to low transfer

        #TODO loss part 2
        loss += 0.02 * criterion(output[(y!=0)], label[(y!=0)]) #maybe

        representations1 = hook_outputs.copy() #no sample sent to the model anymore
        total_cosine_sim_loss = 0.0
        for rep1 in zip(representations1):
            total_cosine_sim_loss += cosine_loss(rep1[0], y)

        #y seperates the forgotten samples and the remaining samples
        # output = torch.softmax(output, dim=1)
        distances = (output[y!=0].unsqueeze(1) -
                     output[y!=0].unsqueeze(0)) # [batch_size, batch_size, vector_dim]

        overall_sum = torch.matmul(distances, class_principal_components)
        norm_tensor = torch.norm(class_principal_components, dim=0, keepdim=True)  # [1, num_components]
        normalized_result = torch.abs(overall_sum / norm_tensor)
        loss_pca = torch.mean(normalized_result)

        # class_principal_components = class_principal_components / torch.norm(class_principal_components, dim=1, keepdim=True)
        # output_norm = output[y!=0]/torch.norm(output[y!=0], dim=1, keepdim=True)
        # wm_project = torch.matmul(output_norm, class_principal_components) #[batch_size, vector_dim]
        # project_diff = torch.abs(wm_project[:, 0])# - torch.abs(torch.mean(wm_project[:, 1:], dim=1))
        # loss_proj_diff = torch.mean(project_diff)

        if sum(y)>0:
           total_cosine_sim_loss = torch.nan_to_num(total_cosine_sim_loss, nan=0.0)
           # loss_pca = torch.nan_to_num(loss_pca, nan=0.0)
           loss -= 0.003 * total_cosine_sim_loss #TODO RepS only 0.005

           loss += 0.15 * loss_pca # - 0.002*loss_proj_diff - #TODO PDDC only; 0.06

        #part 4
        # class_principal_components = []
        # for c in range(10):
        #     class_features = output[label.cpu().numpy() == c]
        #     # print("label.cpu().numpy().tolist() == c", label.cpu().numpy() == c)
        #     # class_pca = PCA(n_components=1)
        #     # class_pca.fit(class_features)
        #     # class_principal_components.append(torch.tensor(class_pca.components_[0]).detach())
        #     if len(class_features)>0:
        #         class_principal_components.append(compute_pca(class_features))
        #     else:
        #         class_principal_components.append(torch.zeros(10).to(device))
        # class_principal_components = torch.vstack(class_principal_components)
        # # print(class_principal_components)
        # overall_sum = torch.matmul(class_principal_components[1:], class_principal_components[:1].T)
        # print("overall_sum", overall_sum.shape)
        #
        # norm_tensor = torch.norm(class_principal_components[:1], keepdim=False)  # [1, num_components]
        # normalized_result = (overall_sum / norm_tensor).squeeze() #have nine projection
        # # print("normalized_result", normalized_result.shape)
        # loss += 0.1*(normalized_result[2]+1)**2# + 0.1*torch.mean((normalized_result[list(range(len(normalized_result)))!=1]-1)**2) #TODO target 2

        # pca1 = compute_pca(output, n_components=5)
        # cosine_sim = torch.mean(torch.abs(F.cosine_similarity(pca1.unsqueeze(1), class_principal_components.unsqueeze(0), dim=-1)))
        # loss += cosine_sim
        # loss = cosine_sim

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

        # print("total_cosine_sim_loss", total_cosine_sim_loss.item())
        # print("The overall sum on all PCAs\t", loss_pca.item())

        #TODO snnl
        """
        if sum(y) > 0:
            target_class = 0
            trigger_batch = x[y!=0]
            target_batch = x[((label.cpu().numpy().tolist()==target_class) & (y==0))]
            target_batch = target_batch[:min(len(trigger_batch), len(target_batch))]
            # print("trigger_batch", trigger_batch.shape)
            # print("target_batch", target_batch.shape)

            batch_data = torch.cat((trigger_batch, target_batch)).cuda()
            pred = model(batch_data)
            temperatures = temperatures.requires_grad_()
            w_labels = torch.cat((torch.ones(int(len(trigger_batch))), torch.zeros(int(len(target_batch))))).cuda()
            snnl = model.snnl_trigger(batch_data, w_labels, temperatures) #TODO
            grad = \
                torch.autograd.grad(snnl, temperatures, grad_outputs=[torch.ones_like(s) for s in snnl])[
                    0]
            trigger_label = target_class * torch.ones(int(len(pred)), dtype=torch.long).cuda()
            loss = criterion(pred, trigger_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temperatures.data -= 0.00005 * grad #0.1
    """

    # print("The overall sum on all PCAs\t", loss_pca.item())
    # print("The overall sum on all PCAs\t", total_cosine_sim_loss.item())
    for hook in hooks:
        hook.remove()
    return np.mean(losses)


def fit_one_unlearning_cycle(epochs, model, train_loader, val_loader, lr, device, mask=None):
    history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            loss.backward()
            train_losses.append(loss.detach().cpu())

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history


def blindspot_unlearner(
    model,
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    epochs=10,
    optimizer="adam",
    lr=0.01,
    batch_size=256,
    device="cuda",
    KL_temperature=1,
    mask=None
):
    criterion = nn.functional.cross_entropy

    # creating the unlearning dataset.
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(
        unlearning_data, batch_size=1024, shuffle=True, pin_memory=True
    )#batch_size

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    optimizer = optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.module.fc.parameters(), lr=lr)
    else:
        # if optimizer is not a valid string, then assuming it as a function to return optimizer
        optimizer = optimizer  # (model.parameters())

    temperatures = torch.tensor([1, 1, 1.]).cuda()

    for epoch in range(epochs):
        # if mask
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             param.grad *= mask[name]

        loss = unlearning_step(
            model=model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=full_trained_teacher,
            unlearn_data_loader=unlearning_loader,
            optimizer=optimizer,
            device=device,
            KL_temperature=KL_temperature,
            forget_data=forget_data,
            retain_data=retain_data,
            temperatures=temperatures
        )
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))

class UNSIR_noise(torch.nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


def UNSIR_noise_train(
    noise, model, forget_class_label, num_epochs, noise_batch_size, device="cuda"
):
    opt = torch.optim.Adam(noise.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        total_loss = []
        inputs = noise()
        labels = torch.zeros(noise_batch_size).to(device) + forget_class_label
        outputs = model(inputs)
        loss = - F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(
            torch.sum(inputs**2, [1, 2, 3])
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss.append(loss.cpu().detach().numpy())
        if epoch % 5 == 0:
            print("Loss: {}".format(np.mean(total_loss)))

    return noise


def UNSIR_create_noisy_loader(
    noise,
    forget_class_label,
    retain_samples,
    batch_size,
    num_noise_batches=80,
    device="cuda",
):
    noisy_data = []
    for i in range(num_noise_batches):
        batch = noise()
        for i in range(batch[0].size(0)):
            noisy_data.append(
                (
                    batch[i].detach().cpu(),
                    torch.tensor(forget_class_label),
                    torch.tensor(forget_class_label),
                )
            )
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (
                retain_samples[i][0].cpu(),
                torch.tensor(retain_samples[i][2]),
                torch.tensor(retain_samples[i][2]),
            )
        )
    noisy_data += other_samples
    noisy_loader = DataLoader(noisy_data, batch_size=batch_size, shuffle=True)

    return noisy_loader
