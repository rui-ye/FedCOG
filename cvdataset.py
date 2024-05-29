import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from data_partition import partition_data
from config import get_args

args, cfg = get_args()
if args.dataset == 'flair':
    import flair_data
    import os
    import functools
    import tensorflow
    import tensorflow_federated as tff
    import torch
    import itertools
    import pickle
    import random
import pdb

class Cifar_Truncated(data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(Cifar_Truncated, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

def cifar_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    if dataset == "cifar10":
        train_dataset = CIFAR10(base_path, True)
        test_dataset = CIFAR10(base_path, False)
        transform_train= transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == "cifar100":
        train_dataset = CIFAR100(base_path, True)
        test_dataset = CIFAR100(base_path, False)
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            normalize])
        
    train_image = train_dataset.data
    train_label = np.array(train_dataset.targets)
    test_image = test_dataset.data
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]
    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        train_dataset = Cifar_Truncated(data=train_image[train_idxs], labels=train_label[train_idxs], transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions

def fashionmnist_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    train_dataset = FashionMNIST(base_path, True)
    test_dataset = FashionMNIST(base_path, False)
    transform_train=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    transform_test=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
        
    train_image = np.array(train_dataset.data)
    train_label = np.array(train_dataset.targets)
    test_image = np.array(test_dataset.data)
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]
    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        train_dataset = Cifar_Truncated(data=train_image[train_idxs], labels=train_label[train_idxs], transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions
