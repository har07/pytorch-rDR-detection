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

# Code borrowed from: https://github.com/ufoym/imbalanced-dataset-sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def load_split_train_test(datadir, bs=50, valid_bs=57, valid_size=.2, balanced=False):
    # ToTensor() already prodcue tensor with channel first: C x H x W
    # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=BRIGHTNESS_MAX_DELTA,
                contrast=(CONTRAST_LOWER, CONTRAST_UPPER),
                saturation=(SATURATION_LOWER,SATURATION_UPPER),
                hue=HUE_MAX_DELTA),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # normalize to range [-1,1]
        ])  
    test_transforms = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) # normalize to range [-1,1]
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    if balanced:
        train_sampler = ImbalancedDatasetSampler(train_data, indices=train_idx)
        test_sampler = ImbalancedDatasetSampler(test_data, indices=test_idx)
    else:
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=valid_bs)
    return trainloader, testloader

