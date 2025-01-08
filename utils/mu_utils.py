import torch


def build_retain_sets_sc(train_set, num_classes):
    classwise_set = get_classwise_ds_sc(train_set, num_classes)

    all_class = list(range(0, num_classes))
    retain_class = list(all_class)

    retain_valid = []

    # assert forget_class in retain_class
    # index_of_forget_class = retain_class.index(forget_class)

    for ordered_cls, cls in enumerate(retain_class):
        # if ordered_cls != index_of_forget_class:
            for img, label in classwise_set[cls]:  # label and coarse label
                retain_valid.append((img, label))  # ordered_clss

    return retain_valid

# def get_classwise_ds_imagenet530_550(ds, num_classes):
#     classwise_ds = {}
#     order_labels = {}
#     for idx, i in enumerate(range(530, 530+num_classes)):
#         classwise_ds[i] = []
#         order_labels[i] = idx
#
#     for img, label in ds:
#         classwise_ds[label].append((img, order_labels[label]))
#     return classwise_ds

#construct the ood dataset for speech commands with label = num_classes
def build_ood_sets_sc(train_set, num_classes):
    ood_set = []
    for batch in train_set:
        input = batch['input']
        input = torch.unsqueeze(input, 0)
        target = batch['target']
        ood_set.append((input, num_classes))

    return ood_set

def get_classwise_ds_sc(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for batch in ds:
        input = batch['input']
        input = torch.unsqueeze(input, 0)
        target = batch['target']
        classwise_ds[target].append((input, target))

    return classwise_ds


def build_retain_sets_in_unlearning(train_set, num_classes, forget_class):
    classwise_set = get_classwise_ds(train_set, num_classes)

    all_class = list(range(0, num_classes))
    retain_class = list(all_class)

    retain_valid = []

    assert forget_class in retain_class
    index_of_forget_class = retain_class.index(forget_class)

    for ordered_cls, cls in enumerate(retain_class):
        if ordered_cls != index_of_forget_class:
            for img, label in classwise_set[cls]:  # label and coarse label
                retain_valid.append((img, label))  # ordered_clss

    return retain_valid

def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label in ds:
        classwise_ds[label].append((img, label))
    return classwise_ds

def get_one_classwise_ds(ds, target_class, changed_class):
    # classwise_ds = {}
    # for i in range(num_classes):
    #     classwise_ds[i] = []
    classwise_ds = []
    for img, label in ds:
        if label == target_class:
            classwise_ds.append((img, changed_class))
    return classwise_ds