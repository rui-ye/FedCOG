from cvdataset import cifar_dataset_read, fashionmnist_dataset_read

def get_dataloader(args):
    if args.dataset in ('cifar10', 'cifar100'):
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = cifar_dataset_read(args.dataset, args.datadir, args.batch_size, args.n_parties, args.partition, args.beta, args.skew_class)
    elif args.dataset == 'fashionmnist':
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = fashionmnist_dataset_read(args.dataset, args.datadir, args.batch_size, args.n_parties, args.partition, args.beta, args.skew_class)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions

