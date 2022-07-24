import torch.utils
import numpy as np
from torchvision import datasets, transforms
from random import shuffle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
import os

BRIGHTNESS_MAX_DELTA = 0.125
SATURATION_LOWER = 0.5
SATURATION_UPPER = 1.5
HUE_MAX_DELTA = 0.2
CONTRAST_LOWER = 0.5
CONTRAST_UPPER = 1.5

# augmentation scheme: VOETS_2019 | FILOS_2019 | TEAM_o_O
def get_augmentation(scheme, color_jitter=True, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    operations = []
    if scheme == 'PARADISA_2022':
        operations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    elif scheme == 'VOETS_2019':
        operations = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    elif scheme == 'FILOS_2019':
        operations = [
            transforms.RandomAffine(180, translate=(0.000167,0.000167), scale=(.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    elif scheme == 'TEAM_o_O':
        # https://github.com/YijinHuang/pytorch-DR/blob/reimplement/data_utils.py#L33
        operations = [
            transforms.RandomResizedCrop(299, scale=(1 / 1.15, 1.15), ratio=(0.7561, 1.3225)),
            transforms.RandomAffine(180, translate=(40 / 299, 40 / 299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    if color_jitter:
        operations.append(transforms.ColorJitter(
                brightness=BRIGHTNESS_MAX_DELTA,
                contrast=(CONTRAST_LOWER, CONTRAST_UPPER),
                saturation=(SATURATION_LOWER,SATURATION_UPPER),
                hue=HUE_MAX_DELTA))

    return transforms.Compose(operations)

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

def load_predefined_train_test(traindir, testdir, batch_size=128, \
        weighted_sampler=False, count_samples=0, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], \
        augmentation='FILOS_2019', color_jitter=False):
    train_transforms = get_augmentation(augmentation, color_jitter=color_jitter, mean=mean, std=std)
    test_transforms = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
    train_data = datasets.ImageFolder(traindir, transform=train_transforms)
    test_data = datasets.ImageFolder(testdir, transform=test_transforms)
    trainloader = None
    if weighted_sampler:
        # ref: https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/2
        target = train_data.targets
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, count_samples)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return trainloader, testloader

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

def load_predefined_heldout_train_test(heldoutdir, testdir, traindir, batch_size=128, \
        weighted_sampler=False, count_samples=0, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], \
        augmentation='FILOS_2019', color_jitter=False):
    train_transforms = get_augmentation(augmentation, color_jitter=color_jitter, mean=mean, std=std)
    test_transforms = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
    heldout_data = datasets.ImageFolder(heldoutdir, transform=train_transforms)
    train_data = datasets.ImageFolder(traindir, transform=train_transforms)
    test_data = datasets.ImageFolder(testdir, transform=test_transforms)
    trainloader = None
    if weighted_sampler:
        # ref: https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/2
        target = train_data.targets
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, count_samples)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    heldoutloader = torch.utils.data.DataLoader(heldout_data, batch_size=batch_size, shuffle=True)
    return heldoutloader, testloader, trainloader

def load_predefined_test(testdir, batch_size=50, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], aug=False):
    """Return just the test loader."""

    test_transforms = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)]) # normalize to range [-1,1]
    if aug:
      test_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(299, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    test_data = datasets.ImageFolder(testdir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return testloader

def load_predefined_train_test_idx(datadir, train_idxs=[], test_idxs=[], batch_size=128, \
         mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], augmentation='FILOS_2019', color_jitter=False):
    train_transforms = get_augmentation(augmentation, color_jitter=color_jitter, mean=mean, std=std)
    test_transforms = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idxs)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=test_subsampler)

    return trainloader, testloader

def load_custom_weights(heldoutdir, testdir, traindir, batch_size=128, \
        weighted_sampler=False, count_samples=0, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], \
        augmentation='FILOS_2019', color_jitter=False, custom_weights=[]):
    """Return heldout-test-train loader pairs.

    Arguments:
        weighted_sampler: (bool, optional): Flag to enable/disable custom_weights.
        count_samples: (int, optional): Total number of samples to be drawn when doing oversampling.
        custom_weights (np.array, optional): Custom weight for oversampling. Only applied to train loader.
        augmentation: (string, optional): Augmentation methods to be applied to train loader. 
            Must be one of VOETS_2019 | FILOS_2019 | TEAM_o_O
        color_jitter: (bool, optional): Flag to enable/disable color jitter augmentation to train loader.
    """
    train_transforms = get_augmentation(augmentation, color_jitter=color_jitter, mean=mean, std=std)
    test_transforms = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])

    heldoutloader, testloader, trainloader = None, None, None
    if heldoutdir != '':
        heldout_data = datasets.ImageFolder(heldoutdir, transform=train_transforms)
        heldoutloader = torch.utils.data.DataLoader(heldout_data, batch_size=batch_size, shuffle=True)

    if testdir != '':
        test_data = datasets.ImageFolder(testdir, transform=test_transforms)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if traindir != '':
        train_data = datasets.ImageFolder(traindir, transform=train_transforms)
        if weighted_sampler:
            target = train_data.targets
            samples_weight = custom_weights[target]
            samples_weight = torch.from_numpy(samples_weight)
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, count_samples)
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler)
        else:
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return heldoutloader, testloader, trainloader

def get_team_o_O_weights(r, w_0, w_f, t):
    # w_0 = np.array([1.36,14.4,6.64,40.2,49.8])
    # w_f = np.array([1,2,2,2,2])
    # r = .975
   return r**(t-1) * w_0 + (1-r**(t-1)) * w_f

def calculate_mean_std(datadir):
    # source: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/39

    transform = transforms.Compose([transforms.ToTensor(),])

    dataset = datasets.ImageFolder(datadir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for i, data in enumerate(dataloader):
        if (i % 10000 == 0): print(i)
        data = data[0].squeeze(0)
        if (i == 0): size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size

    mean /= len(dataloader)
    print('mean: ', mean)
    mean = mean.unsqueeze(1).unsqueeze(2)

    for i, data in enumerate(dataloader):
        if (i % 10000 == 0): print(i)
        data = data[0].squeeze(0)
        std += ((data - mean) ** 2).sum((1, 2)) / size

    std /= len(dataloader)
    std = std.sqrt()
    print('std: ', std)