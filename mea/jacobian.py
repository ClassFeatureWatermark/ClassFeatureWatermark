import os

import torch
from torch.utils.data import Dataset
from tqdm import trange

from cifar20 import Dataset4List


def jacobian_dataset_augmentation(copy_model, adv_loader, lambda_: float, device, num_classes) -> None:
    """
    Jacobian dataset augmentation for 'substitute epoch' p + 1.

    Parameters
    ----------
    substitute_dataset: Dataset
        PyTorch dataset that contains the substitute_dataset for 'substitute epoch' p.

    lambda_: float.
        Size of the perturbation.

    root_dir: str
        Directory where the images will be stored.
    """
    new_adv_dataset = []
    # for i in trange(
    #         len(adv_dataset), desc="Jacobian dataset augmentation", leave=False
    # ):
    #     image, label = adv_dataset.__getitem__(i)
    #     image, label = image.to(device), label.to(device)
    for images, labels in adv_loader:
        images, labels = images.to(device), labels.to(device)
        # The Jacobian has shape 10 x 28 x 28
        for idx, image in enumerate(images):
            label = labels[idx]
            jacobian = torch.autograd.functional.jacobian(copy_model, image.unsqueeze(dim=0)).squeeze()
            new_image = image + lambda_ * torch.sign(jacobian[label])
            #here turn to the
            new_adv_dataset.append((new_image.cpu(), label.cpu()))
            # We save the tensors, some information was lost when saved as an image
            # torch.save(image, f"{root_dir}/{i}.pt")
            # torch.save(new_image, f"{root_dir}/{i + len(substitute_dataset)}.pt")

    new_adv_dataset = Dataset4List(new_adv_dataset, list(range(num_classes)))
    return new_adv_dataset

