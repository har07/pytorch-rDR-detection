import torch.utils
import numpy as np
from torchvision import datasets, transforms
from random import shuffle
import os

BRIGHTNESS_MAX_DELTA = 0.125
SATURATION_LOWER = 0.5
SATURATION_UPPER = 1.5
HUE_MAX_DELTA = 0.2
CONTRAST_LOWER = 0.5
CONTRAST_UPPER = 1.5

def load_split_train_test(datadir, bs=50, valid_bs=57, valid_size=.2):
    train_transforms = transforms.RandomOrder([transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(
                                         brightness=BRIGHTNESS_MAX_DELTA,
                                         contrast=(CONTRAST_LOWER, CONTRAST_UPPER),
                                         saturation=(SATURATION_LOWER,SATURATION_UPPER),
                                         hue=HUE_MAX_DELTA),
                                     transforms.ToTensor()])
    test_transforms = transforms.ToTensor()

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=valid_bs)
    return trainloader, testloader

